# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AdaGO.

Implementation of AdaGO from "AdaGrad Meets Muon: Adaptive Stepsizes for
Orthogonal Updates" (https://arxiv.org/pdf/2509.02981).
"""

import functools
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from optax.contrib import _muon
from optax.contrib import _normuon
from optax.transforms import _masking
import optax.tree
from optax.contrib._cwd import add_cautious_weight_decay

WeightDimNumOrFn = _muon.WeightDimNumOrFn

_is_weight_dim_nums = lambda x: isinstance(x, _muon.MuonDimensionNumbers)

_DEFAULT_NS_COEFFS = _muon._DEFAULT_NS_COEFFS


class AdaGOState(NamedTuple):
  """State for the AdaGO algorithm."""
  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  v_sq: base.Updates
  ns_coeffs: jax.typing.ArrayLike  # shape=(3,) or (n, 3).
  normuon_v: base.Updates | None


class _NormuonUpdateAndV(NamedTuple):
  update: jax.Array
  v: jax.Array


def scale_by_adago(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
        tuple[tuple[jax.typing.ArrayLike, jax.typing.ArrayLike,
                    jax.typing.ArrayLike], ...],
    ] = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    orthogonalization_eps: jax.typing.ArrayLike = 1e-8,
    gamma: jax.typing.ArrayLike = 1.0,
    min_step_size: jax.typing.ArrayLike = 1e-6,
    initial_accumulator_value: jax.typing.ArrayLike = 1.0,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = False,
    use_normuon: bool = False,
    normuon_b2: jax.typing.ArrayLike = 0.95,
    normuon_eps: jax.typing.ArrayLike = 1e-8,
    normuon_rms_scale: jax.typing.ArrayLike = 0.2,
    normuon_v_dtype: Optional[jax.typing.DTypeLike] = None,
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformation:
  r"""Rescale updates according to the AdaGO algorithm.

  AdaGO combines Muon-style orthogonalized momentum with AdaGrad-Norm
  stepsizes. Each parameter keeps a scalar accumulator v_sq updated with
  min(||g||, gamma)^2 and scales the orthogonalized momentum by
  max(min_step_size, lr * min(||g||, gamma) / sqrt(v_sq)).

  Args:
    learning_rate: Global learning rate or schedule.
    ns_coeffs: Coefficients for the Newton-Schulz method.
    ns_steps: Number of Newton-Schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Momentum coefficient (mu in the paper).
    orthogonalization_eps: Term added to denominators in orthogonalization.
    gamma: Clipping threshold for the gradient norm.
    min_step_size: Lower bound for the adaptive step size (epsilon in paper).
    initial_accumulator_value: Initial value v0 for the accumulator.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    use_normuon: Whether to use NorMuon-style neuron-wise normalization.
    normuon_b2: Decay rate for the NorMuon second moment accumulator.
    normuon_eps: Term added inside square roots for NorMuon normalization.
    normuon_rms_scale: Target RMS for NorMuon-normalized updates.
    normuon_v_dtype: Data type for NorMuon second moment accumulator.
    weight_dimension_numbers: Optional tree or callable defining reshape specs.
    consistent_rms: Optional float to activate consistent RMS scaling.
      Ignored when `use_normuon=True`.

  Returns:
    A `GradientTransformation` object.

  References:
    Zhang et al., `AdaGrad Meets Muon: Adaptive Stepsizes for Orthogonal
    Updates <https://arxiv.org/pdf/2509.02981>`_, 2025
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  normuon_v_dtype = utils.canonicalize_dtype(
      jnp.float32 if normuon_v_dtype is None else normuon_v_dtype
  )

  def _resolve_dim_nums(tree):
    if callable(weight_dimension_numbers):
      resolved = weight_dimension_numbers(tree)
    else:
      resolved = weight_dimension_numbers
    if resolved is None or _is_weight_dim_nums(resolved):
      dim_nums = _muon.MuonDimensionNumbers() if resolved is None else resolved
      return jax.tree.map(
          lambda g: dim_nums if g is not None else None,
          tree,
          is_leaf=lambda x: x is None,
      )
    return resolved

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)

    def _init_v_sq(p):
      dtype = jnp.asarray(p).dtype
      return jnp.asarray(initial_accumulator_value**2, dtype=dtype)

    v_sq = jax.tree.map(_init_v_sq, params)
    ns_coeffs_ = jnp.asarray(ns_coeffs)
    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(
          f'ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}'
      )
    normuon_v = None
    if use_normuon:
      resolved_dim_nums = _resolve_dim_nums(params)

      def _init_v_leaf(
          p: jax.Array, dim_num: _muon.MuonDimensionNumbers | None
      ):
        if dim_num is None:
          if p.ndim != 2:
            raise ValueError(
                'NorMuon requires `weight_dimension_numbers` for non-2D tensors'
                f', got rank={p.ndim} and {dim_num=}.'
            )
          dim_num = _muon.MuonDimensionNumbers()
        batch_size, output_size = _normuon._v_shape(p, dim_num)  # pylint: disable=protected-access
        return jnp.zeros((batch_size, output_size), dtype=normuon_v_dtype)

      normuon_v = jax.tree.map(
          _init_v_leaf,
          params,
          resolved_dim_nums,
          is_leaf=_is_weight_dim_nums,
      )
    return AdaGOState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        v_sq=v_sq,
        ns_coeffs=ns_coeffs_,
        normuon_v=normuon_v,
    )

  def update_fn(updates, state, params=None):
    grads = updates
    resolved_weight_dim_nums = _resolve_dim_nums(updates)

    def _leaf_norm(g):
      if g is None:
        return None
      return jnp.linalg.norm(g)

    grad_norms = jax.tree.map(_leaf_norm, grads)

    mu = optax.tree.update_moment(grads, state.mu, beta, 1)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: beta * m + (1 - beta) * g,
          optax.tree.bias_correction(
              mu, beta, numerics.safe_increment(count_inc)
          ),
          optax.tree.bias_correction(grads, beta, count_inc),
      )
    else:
      mu_hat = optax.tree.bias_correction(mu, beta, count_inc)

    def _orthogonalize(x, dim_num):
      if x is None:
        return None
      return _muon.orthogonalize_via_newton_schulz(
          x,
          state.ns_coeffs,
          ns_steps,
          orthogonalization_eps,
          dim_num,
      )

    updates = jax.tree.map(
        _orthogonalize,
        mu_hat,
        resolved_weight_dim_nums,
        is_leaf=lambda x: x is None or _is_weight_dim_nums(x),
    )
    normuon_v = state.normuon_v
    if use_normuon:
      def _normalize_leaf(
          o: jax.Array,
          v: jax.Array,
          dim_num: _muon.MuonDimensionNumbers | None,
      ):
        if dim_num is None:
          if o.ndim != 2:
            raise ValueError(
                'NorMuon requires `weight_dimension_numbers` for non-2D tensors'
                f', got rank={o.ndim} and {dim_num=}.'
            )
          dim_num = _muon.MuonDimensionNumbers()

        reshape_fn, inverse_fn = _muon._compute_muon_reshape(o, dim_num)  # pylint: disable=protected-access
        o_flat = reshape_fn(o)
        mean_sq = jnp.mean(jnp.square(o_flat), axis=1)
        v_new = normuon_b2 * v + (1.0 - normuon_b2) * mean_sq
        denom = jnp.sqrt(v_new[:, None, :] + normuon_eps)
        o_norm = o_flat / denom
        rms = jnp.sqrt(jnp.mean(jnp.square(o_norm), axis=(1, 2)))
        scale = normuon_rms_scale / (rms + normuon_eps)
        o_scaled = o_norm * scale[:, None, None]
        return _NormuonUpdateAndV(inverse_fn(o_scaled), v_new)

      updates_and_v = jax.tree.map(
          _normalize_leaf,
          updates,
          normuon_v,
          resolved_weight_dim_nums,
          is_leaf=_is_weight_dim_nums,
      )
      _is_update_and_v = lambda x: isinstance(x, _NormuonUpdateAndV)
      updates = jax.tree.map(
          lambda uv: uv.update, updates_and_v, is_leaf=_is_update_and_v
      )
      normuon_v = jax.tree.map(
          lambda uv: uv.v, updates_and_v, is_leaf=_is_update_and_v
      )
    else:
      if consistent_rms is not None:
        scaling_fn = functools.partial(
            _muon._scale_update_for_consistent_rms,
            consistent_rms=consistent_rms,
        )
      else:
        scaling_fn = _muon._scale_update_for_width_transfer

      def _scale_update(update, dim_num):
        if update is None:
          return None
        return scaling_fn(update, dim_num)

      updates = jax.tree.map(
          _scale_update,
          updates,
          resolved_weight_dim_nums,
          is_leaf=lambda x: x is None or _is_weight_dim_nums(x),
      )
    lr_t = (
        learning_rate(state.count)
        if callable(learning_rate)
        else learning_rate
    )

    def _clip_norm(n):
      if n is None:
        return None
      gamma_t = jnp.asarray(gamma, dtype=n.dtype)
      return jnp.minimum(n, gamma_t)

    clipped_norms = jax.tree.map(_clip_norm, grad_norms)

    def _update_v_sq(v_sq, n):
      if n is None:
        return v_sq
      return v_sq + n * n

    new_v_sq = jax.tree.map(_update_v_sq, state.v_sq, clipped_norms)

    def _step_size(n, v_sq):
      if n is None:
        return None
      lr = jnp.asarray(lr_t, dtype=n.dtype)
      min_step = jnp.asarray(min_step_size, dtype=n.dtype)
      lr_safe = jnp.where(lr > 0, lr, jnp.asarray(1.0, dtype=n.dtype))
      return jnp.maximum(min_step / lr_safe, n / jnp.sqrt(v_sq))

    step_sizes = jax.tree.map(_step_size, clipped_norms, new_v_sq)

    def _apply_step(u, step):
      if u is None:
        return None
      if step is None:
        return None
      return step * u

    updates = jax.tree.map(_apply_step, updates, step_sizes)

    mu = optax.tree.cast(mu, mu_dtype)
    if normuon_v is not None:
      normuon_v = optax.tree.cast(normuon_v, normuon_v_dtype)
    return updates, AdaGOState(
        count=count_inc,
        mu=mu,
        v_sq=new_v_sq,
        ns_coeffs=state.ns_coeffs,
        normuon_v=normuon_v,
    )
  return base.GradientTransformation(init_fn, update_fn)


def adago(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
        tuple[tuple[jax.typing.ArrayLike, jax.typing.ArrayLike,
                    jax.typing.ArrayLike], ...],
    ] = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    orthogonalization_eps: jax.typing.ArrayLike = 1e-8,
    gamma: jax.typing.ArrayLike = 1.0,
    min_step_size: jax.typing.ArrayLike = 1e-6,
    initial_accumulator_value: jax.typing.ArrayLike = 1.0,
    weight_decay: base.ScalarOrSchedule = 0.0,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    cautious_weight_decay: bool = False,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = False,
    use_normuon: bool = False,
    normuon_b2: jax.typing.ArrayLike = 0.95,
    normuon_eps: jax.typing.ArrayLike = 1e-8,
    normuon_rms_scale: jax.typing.ArrayLike = 0.2,
    normuon_v_dtype: Optional[jax.typing.DTypeLike] = None,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: jax.typing.ArrayLike = 0.0,
    adago_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformation:
  r"""AdaGO optimizer with Muon/NorMuon-style orthogonalization.

  AdaGO applies orthogonalized momentum updates to matrix parameters and uses
  an AdaGrad-Norm style stepsize with clipping and a minimum step size. When
  `use_normuon=True`, the orthogonalized updates are normalized using
  NorMuon-style neuron-wise second moments. Non-2D parameters are optimized
  with AdamW, mirroring the Muon setup.

  Args:
    learning_rate: Global learning rate or schedule for AdaGO and AdamW.
    ns_coeffs: Coefficients for the Newton-Schulz method.
    ns_steps: Number of Newton-Schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Momentum coefficient for AdaGO (mu in the paper).
    orthogonalization_eps: Term added to denominators in orthogonalization.
    gamma: Clipping threshold for the gradient norm.
    min_step_size: Lower bound for the adaptive step size (epsilon in paper).
    initial_accumulator_value: Initial value v0 for the accumulator.
    weight_decay: Strength of weight decay regularization for AdaGO parameters.
    weight_decay_mask: Mask for weight decay.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    use_normuon: Whether to use NorMuon-style neuron-wise normalization.
    normuon_b2: Decay rate for the NorMuon second moment accumulator.
    normuon_eps: Term added inside square roots for NorMuon normalization.
    normuon_rms_scale: Target RMS for NorMuon-normalized updates.
    normuon_v_dtype: Data type for NorMuon second moment accumulator.
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.
    adago_weight_dimension_numbers: Optional tree of
      `_muon.MuonDimensionNumbers`s
      specifying how to reshape the parameters for orthogonalization; a `None`
      value indicates the parameter is not an AdaGO parameter and will be
      optimized with AdamW. A callable takes as input the params and returns a
      possibly masked pytree of specs, similar to `weight_decay_mask`. If not
      provided, AdaGO is applied to all 2D parameters.
    consistent_rms: Optional float to activate consistent RMS scaling.
      If `None`, uses width scaling `sqrt(max(1, fan_out / fan_in))`. This is
      ignored when `use_normuon=True`.

  Returns:
    The corresponding `GradientTransformation`.

  References:
    Zhang et al., `AdaGrad Meets Muon: Adaptive Stepsizes for Orthogonal
    Updates <https://arxiv.org/pdf/2509.02981>`_, 2025
  """
  # None at root indicates the default 2D rule.
  if adago_weight_dimension_numbers is None:
    param_labels = lambda params: jax.tree.map(
        lambda x: 'adago' if x.ndim == 2 else 'adam', params
    )
    adago_weight_dimension_numbers = _muon.MuonDimensionNumbers()
  else:
    def param_labels(params):
      dim_nums = (adago_weight_dimension_numbers(params)
                  if callable(adago_weight_dimension_numbers)
                  else adago_weight_dimension_numbers)
      populate_subtree_ = lambda dim_num, x: jax.tree.map(
          lambda y: 'adago' if dim_num is not None else 'adam', x)
      # Dimension numbers come first since they can be a prefix mask.
      return jax.tree.map(
          populate_subtree_,
          dim_nums,
          params,
          is_leaf=lambda x: x is None or _is_weight_dim_nums(x),
      )

  # Normalize dimension numbers to match the masked AdaGO state tree.
  def adago_weight_dim_nums_fn(params):
    dim_nums = (adago_weight_dimension_numbers(params)
                if callable(adago_weight_dimension_numbers)
                else adago_weight_dimension_numbers)
    mask = jax.tree.map(lambda label: label == 'adago', param_labels(params))
    is_leaf = lambda x: (x is None or _is_weight_dim_nums(x)
                         or isinstance(x, _masking.MaskedNode))
    populate_subtree_ = lambda dim_nums, submask: jax.tree.map(
        lambda m: dim_nums if m else _masking.MaskedNode(), submask)
    return jax.tree.map(populate_subtree_, dim_nums, mask, is_leaf=is_leaf)

  return combine.partition(
      transforms={
          'adago': combine.chain(
              scale_by_adago(
                  learning_rate=learning_rate,
                  ns_coeffs=ns_coeffs,
                  ns_steps=ns_steps,
                  beta=beta,
                  orthogonalization_eps=orthogonalization_eps,
                  gamma=gamma,
                  min_step_size=min_step_size,
                  initial_accumulator_value=initial_accumulator_value,
                  mu_dtype=mu_dtype,
                  nesterov=nesterov,
                  use_normuon=use_normuon,
                  normuon_b2=normuon_b2,
                  normuon_eps=normuon_eps,
                  normuon_rms_scale=normuon_rms_scale,
                  normuon_v_dtype=normuon_v_dtype,
                  weight_dimension_numbers=adago_weight_dim_nums_fn,
                  consistent_rms=consistent_rms,
              ),
              (
                  add_cautious_weight_decay(weight_decay, weight_decay_mask)
                  if cautious_weight_decay
                  else transform.add_decayed_weights(
                      weight_decay, weight_decay_mask
                  )
              ),
              transform.scale_by_learning_rate(learning_rate),
          ),
          'adam': alias.adamw(
              learning_rate=learning_rate,
              b1=adam_b1,
              b2=adam_b2,
              eps=orthogonalization_eps,
              eps_root=adam_eps_root,
              weight_decay=adam_weight_decay,
              cautious_weight_decay=cautious_weight_decay,
              mu_dtype=mu_dtype,
              nesterov=nesterov,
          ),
      },
      param_labels=param_labels,
  )
