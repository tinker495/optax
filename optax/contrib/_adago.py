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
import math
from typing import Any, Callable, NamedTuple, Optional, Union, Sequence

import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from optax.transforms import _masking
import optax.tree

ReshapeFn = Callable[[jax.Array], jax.Array]


_DEFAULT_NS_COEFFS = (3.4445, -4.7750, 2.0315)


class MuonDimensionNumbers(NamedTuple):
  """Specification for which weight axes participate in matrix projection.

  Muon/AdaGO define an orthogonalization for 2D matrix weights for
  matrix-vector products:

  .. math::
    x W = y

  where the first matrix dimension is the reduction axis and the second matrix
  dimension is the output axis. Thus, the default spec consists of 0 and 1
  reduction and output axes respectively.

  .. warning::
    The batch axes are implicit, all axes not specified as reduction or output
    axes are considered batch axes and will be considered independently in the
    orthogonalization (via jax.vmap).
  """
  reduction_axis: Sequence[int] | int = 0
  output_axis: Sequence[int] | int = 1

WeightDimNumOrFn = MuonDimensionNumbers | base.Params | Callable[
    [base.Params], base.Params | None]


_is_weight_dim_nums = lambda x: isinstance(x, MuonDimensionNumbers)


def _normalize_axes(x: jax.Array, dim_nums: MuonDimensionNumbers) -> tuple[
    tuple[int, ...], tuple[int, ...]]:
  """Normalize axes in dimension numbers to two tuples of non-negative ints."""
  if isinstance(dim_nums.reduction_axis, int):
    dim_nums = dim_nums._replace(reduction_axis=(dim_nums.reduction_axis,))
  reduction_axes = tuple(ax % x.ndim for ax in dim_nums.reduction_axis)

  if isinstance(dim_nums.output_axis, int):
    dim_nums = dim_nums._replace(output_axis=(dim_nums.output_axis,))
  output_axes = tuple(ax % x.ndim for ax in dim_nums.output_axis)
  return reduction_axes, output_axes


def _compute_muon_reshape(x: jax.Array, dim_nums: MuonDimensionNumbers
                          ) -> tuple[ReshapeFn, ReshapeFn]:
  """Compute the reshape and inverse functions for an array from a spec."""
  if x.ndim < 2:
    raise ValueError('Muon optimized parameters must have rank >= 2, got'
                     f' {x.ndim=}')
  reduction_axes, output_axes = _normalize_axes(x, dim_nums)
  if set(reduction_axes) & set(output_axes):
    raise ValueError('Normalized reduction axes and output axes must be'
                     f' disjoint, got {reduction_axes} and {output_axes}.'
                     f' Originally {dim_nums=} and {x.shape=}')
  batch_axes = tuple(sorted(set(range(x.ndim)) - set(reduction_axes)
                            - set(output_axes)))
  transpose = batch_axes + reduction_axes + output_axes
  inv_transpose = tuple(sorted(range(x.ndim), key=lambda i: transpose[i]))
  axes2shape = lambda axes: tuple(x.shape[ax] for ax in axes)
  # Reshape to (batch, reduction, output) to match the (reduction, output)
  # structure of the original muon for 2D weights.
  flat_shape = (
      math.prod(axes2shape(batch_axes)),
      math.prod(axes2shape(reduction_axes)),
      math.prod(axes2shape(output_axes)),
  )
  unflat_shape = (
      axes2shape(batch_axes)
      + axes2shape(reduction_axes)
      + axes2shape(output_axes)
  )
  reshape_fn = lambda x: x.transpose(transpose).reshape(flat_shape)
  inverse_fn = lambda x: x.reshape(unflat_shape).transpose(inv_transpose)
  return reshape_fn, inverse_fn


def _get_shape_products(
    x: jax.Array, dim_nums: MuonDimensionNumbers
) -> tuple[float, float]:
  reduction_axes, output_axes = _normalize_axes(x, dim_nums)
  fan_in = math.prod(x.shape[ax] for ax in reduction_axes)
  fan_out = math.prod(x.shape[ax] for ax in output_axes)
  return fan_in, fan_out


def _scale_update_for_width_transfer(
    update: jax.Array, dim_nums: MuonDimensionNumbers
):
  """Apply width scaling from <https://github.com/KellerJordan/Muon>."""
  fan_in, fan_out = _get_shape_products(update, dim_nums)
  scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
  return scale * update


def _scale_update_for_consistent_rms(
    update: jax.Array,
    dim_nums: MuonDimensionNumbers,
    consistent_rms: jax.typing.ArrayLike
):
  """Apply consistent RMS scaling from <https://arxiv.org/abs/2502.16982>."""
  fan_in, fan_out = _get_shape_products(update, dim_nums)
  scale = jnp.sqrt(jnp.maximum(fan_in, fan_out)) * consistent_rms
  return scale * update


def scale_by_shape(
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformation:
  """Scale updates by factors derived from parameter shape.

  Args:
    weight_dimension_numbers: An optional tree with the same structure as the
      params of `MuonDimensionNumbers`s, specifying how to reshape the
      parameters before and after the orthogonalization OR a callable returning
      such a tree. None implies that all parameters are 2D matrices.
    consistent_rms: An optional float to activate consistent RMS scaling.
      If float, scales updates by `sqrt(max(fan_in, fan_out)) * consistent_rms`.
      If None, uses width scaling `sqrt(max(1, fan_out / fan_in))`.

  Returns:
    A `GradientTransformation` object.
  """

  def update_fn(updates, state, params=None):
    del params
    if callable(weight_dimension_numbers):
      # Populate weight_dim_nums if it's a callable. Use updates instead of
      # actual params since only shapes matter and params may not be provided.
      resolved_weight_dim_nums = weight_dimension_numbers(updates)
    else:
      resolved_weight_dim_nums = weight_dimension_numbers

    if consistent_rms is not None:
      scaling_fn = functools.partial(
          _scale_update_for_consistent_rms, consistent_rms=consistent_rms
      )
    else:
      scaling_fn = _scale_update_for_width_transfer

    scaled_updates = jax.tree.map(
        scaling_fn,
        updates,
        resolved_weight_dim_nums,
        is_leaf=_is_weight_dim_nums,
    )
    return scaled_updates, state

  # Use the standard empty_state initializer, as this transform is stateless
  return base.GradientTransformation(base.init_empty_state, update_fn)


def _newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
  # Implements Newton-Schulz step f(X) = c_0 X + c_1 (XX^T)X + c_2 (XX^T)^2X,
  # with quintic form f(X) = c_0 X + (c_1 A + c_2 AA)X, where A = XX^T.
  # The NS step has the property f(X) = f(X^T)^T. That is, we can get equivalent
  # result by transposing input and output. In particular, we may transpose X
  # when rows > cols for efficiency.
  a = x @ x.T.conj()
  b = coeffs[1] * a + coeffs[2] * a @ a
  return coeffs[0] * x + b @ x


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike = 5,
    eps: jax.typing.ArrayLike = 1e-8,
    dimension_numbers: MuonDimensionNumbers | None = None,
) -> jax.Array:
  r"""Orthogonalize via Newton-Schulz iteration.

  We opt to use a quintic iteration whose coefficients are selected to maximize
  the slope at zero. For the purpose of minimizing steps, it turns out to be
  empirically effective to keep increasing the slope at zero even beyond the
  point where the iteration no longer converges all the way to one everywhere
  on the interval. This iteration therefore does not produce UV^T but rather
  something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
  which turns out not to hurt model performance at all relative to UV^T, where
  USV^T = G is the SVD.

  Args:
    x: A matrix to orthogonalize.
    ns_coeffs: Coefficients for the Newton-schulz iterators.
      Must have shape (n, 3) where n is the number of iterations.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a 2D array.
    eps: Term added to denominators to improve numerical stability.
    dimension_numbers: Optional spec for reshaping a tensor before and after the
      orthogonalization, to support non-2D parameters.

  Returns:
    The orthogonalized matrix.
  """
  if x.ndim != 2 and not isinstance(dimension_numbers, MuonDimensionNumbers):
    raise ValueError(
        f'Input must have shape (m, n) or weight dimension numbers must be'
        f' provided. Got shape={x.shape} and {dimension_numbers=}.')
  if x.ndim == 2:
    dimension_numbers = MuonDimensionNumbers(reduction_axis=0, output_axis=1)
  if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
    raise ValueError('Newton-Schulz coefficients must have shape (3,) or'
                     f' (n, 3), got {ns_coeffs.shape}')

  def _orthogonalize(x):
    transposed = False
    if x.shape[0] > x.shape[1]:
      x = x.T
      transposed = True

    x /= jnp.linalg.norm(x) + eps  # Ensure spectral norm is at most 1
    ns_coeffs_ = ns_coeffs.astype(x.dtype)
    if ns_coeffs_.ndim == 1:
      x = jax.lax.fori_loop(
          0, ns_steps, lambda _, x: _newton_schulz_iterator(x, ns_coeffs_), x,
          unroll=True)  # Unroll to ensure efficient composition with jax.vmap.
    else:
      x, _ = jax.lax.scan(
          lambda x, abc: (_newton_schulz_iterator(x, abc), None), x, ns_coeffs_
      )
    if transposed:
      x = x.T
    return x

  reshape_fn, inverse_fn = _compute_muon_reshape(x, dimension_numbers)
  return inverse_fn(jax.vmap(_orthogonalize)(reshape_fn(x)))


class AdaGOState(NamedTuple):
  """State for the AdaGO algorithm."""
  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  v_sq: base.Updates
  ns_coeffs: jax.typing.ArrayLike  # shape=(3,) or (n, 3).


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
    weight_dimension_numbers: Optional tree or callable defining reshape specs.
    consistent_rms: Optional float to activate consistent RMS scaling.

  Returns:
    A `GradientTransformation` object.

  References:
    Zhang et al., `AdaGrad Meets Muon: Adaptive Stepsizes for Orthogonal
    Updates <https://arxiv.org/pdf/2509.02981>`_, 2025
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

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
    return AdaGOState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        v_sq=v_sq,
        ns_coeffs=ns_coeffs_,
    )

  def update_fn(updates, state, params=None):
    grads = updates
    if callable(weight_dimension_numbers):
      # Populate weight_dim_nums if it's a callable. Use updates instead of
      # actual params since only shapes matter and params may not be provided.
      resolved_weight_dim_nums = weight_dimension_numbers(updates)
    elif weight_dimension_numbers is None or _is_weight_dim_nums(
        weight_dimension_numbers
    ):
      dim_nums = (
          MuonDimensionNumbers()
          if weight_dimension_numbers is None
          else weight_dimension_numbers
      )
      resolved_weight_dim_nums = jax.tree.map(
          lambda g: dim_nums if g is not None else None,
          updates,
          is_leaf=lambda x: x is None,
      )
    else:
      resolved_weight_dim_nums = weight_dimension_numbers

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
      return orthogonalize_via_newton_schulz(
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

    if consistent_rms is not None:
      scaling_fn = functools.partial(
          _scale_update_for_consistent_rms, consistent_rms=consistent_rms
      )
    else:
      scaling_fn = _scale_update_for_width_transfer

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
    return updates, AdaGOState(
        count=count_inc,
        mu=mu,
        v_sq=new_v_sq,
        ns_coeffs=state.ns_coeffs,
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
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = False,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: jax.typing.ArrayLike = 0.0,
    adago_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformation:
  r"""AdaGO optimizer with Muon-style orthogonalization.

  AdaGO applies orthogonalized momentum updates to matrix parameters and uses
  an AdaGrad-Norm style stepsize with clipping and a minimum step size. Non-2D
  parameters are optimized with AdamW, mirroring the Muon setup.

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
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.
    adago_weight_dimension_numbers: Optional tree of `MuonDimensionNumbers`s
      specifying how to reshape the parameters for orthogonalization; a `None`
      value indicates the parameter is not an AdaGO parameter and will be
      optimized with AdamW. A callable takes as input the params and returns a
      possibly masked pytree of specs, similar to `weight_decay_mask`. If not
      provided, AdaGO is applied to all 2D parameters.
    consistent_rms: Optional float to activate consistent RMS scaling.
      If `None`, uses width scaling `sqrt(max(1, fan_out / fan_in))`.

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
    adago_weight_dimension_numbers = MuonDimensionNumbers()
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
                  weight_dimension_numbers=adago_weight_dim_nums_fn,
                  consistent_rms=consistent_rms,
              ),
              transform.add_decayed_weights(weight_decay, weight_decay_mask),
              transform.scale_by_learning_rate(learning_rate),
          ),
          'adam': alias.adamw(
              learning_rate=learning_rate,
              b1=adam_b1,
              b2=adam_b2,
              eps=orthogonalization_eps,
              eps_root=adam_eps_root,
              weight_decay=adam_weight_decay,
              mu_dtype=mu_dtype,
              nesterov=nesterov,
          ),
      },
      param_labels=param_labels,
  )
