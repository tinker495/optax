# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""NorPrism.

NorPrism combines Prism's innovation-aware spectral shaping (right
preconditioning in column space) with NorMuon's neuron-wise normalization
(left preconditioning in row/output space).
"""

import math
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax._src import utils
from optax.contrib import _muon
from optax.transforms import _masking
import optax.tree

MuonDimensionNumbers = _muon.MuonDimensionNumbers
WeightDimNumOrFn = _muon.WeightDimNumOrFn

_DEFAULT_NORPRISM_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

_is_weight_dim_nums = lambda x: isinstance(x, MuonDimensionNumbers)


def _v_shape(x: jax.Array, dim_nums: MuonDimensionNumbers) -> tuple[int, int]:
  """Shape of NorPrism's per-neuron second-moment accumulator."""
  reduction_axes, output_axes = _muon._normalize_axes(x, dim_nums)  # pylint: disable=protected-access
  batch_axes = tuple(
      sorted(set(range(x.ndim)) - set(reduction_axes) - set(output_axes))
  )
  batch_size = math.prod(x.shape[ax] for ax in batch_axes) if batch_axes else 1
  output_size = math.prod(x.shape[ax] for ax in output_axes)
  return batch_size, output_size


def _broadcast_dim_nums(
    tree: base.PyTree,
    dim_nums: MuonDimensionNumbers | None,
) -> base.PyTree:
  """Broadcast a single dim spec (or None) to match a pytree structure."""
  return jax.tree.map(lambda _: dim_nums, tree)


def _resolve_dim_nums(
    tree: base.PyTree,
    weight_dimension_numbers: WeightDimNumOrFn | None,
) -> base.PyTree:
  """Resolve weight dimension numbers to a pytree matching `tree`."""
  if callable(weight_dimension_numbers):
    resolved = weight_dimension_numbers(tree)
  else:
    resolved = weight_dimension_numbers

  if resolved is None or _is_weight_dim_nums(resolved):
    return _broadcast_dim_nums(tree, resolved)
  return resolved


def _inverse_sqrt_via_newton_schulz(
    gram: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int,
    eps: jax.typing.ArrayLike,
) -> jax.Array:
  """Approximate `gram^{-1/2}` using a Newton-Schulz polynomial iteration."""
  if gram.ndim != 2:
    raise ValueError(f'Expected a matrix, got {gram.shape=}.')

  trace = jnp.real(jnp.trace(gram))
  trace = jnp.maximum(trace, jnp.asarray(eps, dtype=trace.dtype))
  eye = jnp.eye(gram.shape[0], dtype=gram.dtype)

  a = gram / trace
  p = eye / jnp.sqrt(trace)

  def _step(a, p, coeff_step):
    coeff_step = coeff_step.astype(gram.dtype)
    a2 = a @ a
    f = coeff_step[0] * eye + coeff_step[1] * a + coeff_step[2] * a2
    return a @ (f @ f), p @ f

  ns_coeffs_ = ns_coeffs.astype(gram.dtype)
  if ns_coeffs_.ndim == 1:

    def _body(_, carry):
      a, p = carry
      return _step(a, p, ns_coeffs_)

    a, p = jax.lax.fori_loop(0, ns_steps, _body, (a, p), unroll=True)
  else:

    def _scan_body(carry, coeff_step):
      a, p = carry
      return _step(a, p, coeff_step), None

    (a, p), _ = jax.lax.scan(_scan_body, (a, p), ns_coeffs_)

  del a  # only inverse square root factor is used.
  return p


def _gram_right_preconditioned_momentum(
    momentum: jax.Array,
    innovation: jax.Array,
    gamma: jax.typing.ArrayLike,
    ns_coeffs: jax.Array,
    ns_steps: int,
    eps: jax.typing.ArrayLike,
) -> jax.Array:
  """Apply Prism-style right preconditioning via Gram-space NS iterations."""
  gram = momentum.conj().T @ momentum
  gram += jnp.asarray(gamma, dtype=momentum.dtype) ** 2 * (
      innovation.conj().T @ innovation
  )
  p = _inverse_sqrt_via_newton_schulz(gram, ns_coeffs, ns_steps, eps)
  return momentum @ p


class NorPrismState(NamedTuple):
  """State for the NorPrism algorithm."""

  mu: base.Updates
  v: base.Updates
  ns_coeffs: jax.Array  # shape=(3,) or (n, 3)


class _NorPrismUpdateAndV(NamedTuple):
  update: jax.Array
  v: jax.Array


def scale_by_norprism(
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
        str,
    ] = _DEFAULT_NORPRISM_NS_COEFFS,
    ns_steps: int = 5,
    b1: float = 0.95,
    b2: float = 0.95,
    gamma: float = 1.0,
    eps: float = 1e-8,
    rms_scale: float = 0.2,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    v_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = True,
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformation:
  r"""Rescale updates according to the NorPrism algorithm.

  NorPrism applies a Prism-style innovation-augmented right preconditioning
  computed in Gram space,

  .. math::
    S_t = M_t^\top M_t + \gamma^2 D_t^\top D_t, \quad D_t = G_t - \bar{M}_t,

  followed by a NorMuon-style output-axis second-moment normalization and RMS
  scaling.

  Args:
    ns_coeffs: Coefficients for the Newton-Schulz method.
    ns_steps: Number of Newton-Schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    b1: Decay rate for the first moment accumulator.
    b2: Decay rate for the output-axis second moment accumulator.
    gamma: Innovation scaling in the Prism augmentation.
    eps: Term added inside square roots for numerical stability.
    rms_scale: Target RMS of each preconditioned-and-normalized update matrix.
    mu_dtype: Data type of the first moment accumulator.
    v_dtype: Data type of the output-axis second moment accumulator.
    nesterov: Whether to use Nesterov momentum.
    weight_dimension_numbers: An optional tree with the same structure as
      params, specifying how to reshape tensors before/after preconditioning.
      A callable may be provided to generate this tree from params/updates.
      `None` implies all parameters are rank-2 matrices.

  Returns:
    A `GradientTransformation` object.
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  v_dtype = jnp.float32 if v_dtype is None else v_dtype
  v_dtype = utils.canonicalize_dtype(v_dtype)

  if isinstance(ns_coeffs, str):
    if ns_coeffs != 'dion':
      raise ValueError(
          "NorPrism supports only 'dion' preset when ns_coeffs is a string"
      )
    ns_coeffs = _DEFAULT_NORPRISM_NS_COEFFS

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)
    ns_coeffs_ = jnp.asarray(ns_coeffs)
    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(
          f'ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}'
      )
    if ns_coeffs_.ndim == 2:
      if ns_coeffs_.shape[0] < ns_steps:
        raise ValueError(f'Not enough coeffs to perform {ns_steps} steps')
      ns_coeffs_ = ns_coeffs_[-ns_steps:]

    resolved_dim_nums = _resolve_dim_nums(params, weight_dimension_numbers)

    def _init_v_leaf(p: jax.Array, dim_num: MuonDimensionNumbers | None):
      if dim_num is None:
        if p.ndim != 2:
          raise ValueError(
              'NorPrism requires `weight_dimension_numbers` for non-2D '
              f'tensors, got rank={p.ndim} and {dim_num=}.'
          )
        dim_num = MuonDimensionNumbers()
      batch_size, output_size = _v_shape(p, dim_num)
      return jnp.zeros((batch_size, output_size), dtype=v_dtype)

    v = jax.tree.map(
        _init_v_leaf,
        params,
        resolved_dim_nums,
        is_leaf=_is_weight_dim_nums,
    )

    return NorPrismState(mu=mu, v=v, ns_coeffs=ns_coeffs_)

  def update_fn(updates, state, params=None):
    del params
    resolved_dim_nums = _resolve_dim_nums(updates, weight_dimension_numbers)

    mu = optax.tree.update_moment(updates, state.mu, b1, 1)
    if nesterov:
      momentum = jax.tree.map(
          lambda m, g: b1 * m + (1.0 - b1) * g,
          mu,
          updates,
      )
    else:
      momentum = mu

    def _norprism_leaf(
        m_t: jax.Array,
        ema_m_t: jax.Array,
        g_t: jax.Array,
        v: jax.Array,
        dim_num: MuonDimensionNumbers | None,
    ):
      if dim_num is None:
        if m_t.ndim != 2:
          raise ValueError(
              'NorPrism requires `weight_dimension_numbers` for non-2D '
              f'tensors, got rank={m_t.ndim} and {dim_num=}.'
          )
        dim_num = MuonDimensionNumbers()

      reshape_fn, inverse_fn = _muon._compute_muon_reshape(  # pylint: disable=protected-access
          m_t, dim_num
      )
      m_t_matrix = reshape_fn(m_t)
      innovation_matrix = reshape_fn(g_t - ema_m_t)

      # Prism-style right preconditioning in Gram space to avoid materializing
      # the augmented [M_t; gamma * D_t] matrix.
      o_p = jax.vmap(
          lambda m, d: _gram_right_preconditioned_momentum(
              m,
              d,
              gamma,
              state.ns_coeffs,
              ns_steps,
              eps,
          )
      )(m_t_matrix, innovation_matrix)

      # NorMuon-style output-axis second-moment normalization.
      mean_sq = jnp.mean(jnp.real(o_p.conj() * o_p), axis=1)
      v_new = b2 * v + (1.0 - b2) * mean_sq
      denom = jnp.sqrt(v_new[:, None, :] + eps)
      o_norm = o_p / denom

      rms = jnp.sqrt(jnp.mean(jnp.real(o_norm.conj() * o_norm), axis=(1, 2)))
      scale = rms_scale / (rms + eps)
      o_scaled = o_norm * scale[:, None, None]

      return _NorPrismUpdateAndV(inverse_fn(o_scaled), v_new)

    updates_and_v = jax.tree.map(
        _norprism_leaf,
        momentum,
        mu,
        updates,
        state.v,
        resolved_dim_nums,
        is_leaf=_is_weight_dim_nums,
    )
    _is_update_and_v = lambda x: isinstance(x, _NorPrismUpdateAndV)
    updates = jax.tree.map(
        lambda uv: uv.update,
        updates_and_v,
        is_leaf=_is_update_and_v,
    )
    v = jax.tree.map(
        lambda uv: uv.v,
        updates_and_v,
        is_leaf=_is_update_and_v,
    )

    mu = optax.tree.cast(mu, mu_dtype)
    v = optax.tree.cast(v, v_dtype)
    return updates, NorPrismState(mu=mu, v=v, ns_coeffs=state.ns_coeffs)

  return base.GradientTransformation(init_fn, update_fn)


def norprism(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
        str,
    ] = _DEFAULT_NORPRISM_NS_COEFFS,
    ns_steps: int = 5,
    b1: float = 0.95,
    b2: float = 0.95,
    gamma: float = 1.0,
    eps: float = 1e-8,
    rms_scale: float = 0.2,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    v_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = True,
    adam_b1: float = 0.9,
    adam_b2: float = 0.95,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    norprism_weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformation:
  r"""NorPrism optimizer with Adam fallback for non-matrix parameters.

  NorPrism composes :func:`scale_by_norprism` with decoupled weight decay and
  learning-rate scaling on matrix parameters, and routes non-matrix parameters
  to AdamW.

  Args:
    learning_rate: Learning rate for NorPrism-updated parameters.
    ns_coeffs: Coefficients for the Newton-Schulz method.
    ns_steps: Number of Newton-Schulz iterations.
    b1: Decay rate for the first moment accumulator.
    b2: Decay rate for the output-axis second moment accumulator.
    gamma: Innovation scaling in the Prism augmentation.
    eps: Term added inside square roots for numerical stability.
    rms_scale: Target RMS of each preconditioned-and-normalized update matrix.
    weight_decay: Weight decay factor for NorPrism parameters.
    weight_decay_mask: Optional mask for NorPrism branch weight decay.
    mu_dtype: Data type of the first moment accumulator.
    v_dtype: Data type of the output-axis second moment accumulator.
    nesterov: Whether to use Nesterov momentum.
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.
    adam_learning_rate: Auxiliary learning rate for Adam branch. If `None`,
      defaults to `learning_rate`.
    norprism_weight_dimension_numbers: Optional tree of
      `MuonDimensionNumbers`s specifying how to reshape NorPrism parameters.
      A `None` leaf indicates an Adam parameter.

  Returns:
    The corresponding `GradientTransformation`.
  """
  if adam_learning_rate is None:
    adam_learning_rate = learning_rate

  if norprism_weight_dimension_numbers is None:
    param_labels = lambda params: jax.tree.map(
        lambda x: 'norprism' if x.ndim == 2 else 'adam', params
    )
    norprism_weight_dimension_numbers = MuonDimensionNumbers()
  else:

    def param_labels(params):
      dim_nums = (
          norprism_weight_dimension_numbers(params)
          if callable(norprism_weight_dimension_numbers)
          else norprism_weight_dimension_numbers
      )
      populate_subtree_ = lambda dim_num, x: jax.tree.map(
          lambda _: 'norprism' if dim_num is not None else 'adam', x
      )
      return jax.tree.map(
          populate_subtree_,
          dim_nums,
          params,
          is_leaf=lambda x: x is None or _is_weight_dim_nums(x),
      )

  def norprism_weight_dim_nums_fn(params):
    dim_nums = (
        norprism_weight_dimension_numbers(params)
        if callable(norprism_weight_dimension_numbers)
        else norprism_weight_dimension_numbers
    )
    mask = jax.tree.map(lambda label: label == 'norprism', param_labels(params))
    is_leaf = lambda x: (
        x is None
        or _is_weight_dim_nums(x)
        or isinstance(x, _masking.MaskedNode)
    )
    populate_subtree_ = lambda dim_num, submask: jax.tree.map(
        lambda m: dim_num if m else _masking.MaskedNode(), submask
    )
    return jax.tree.map(populate_subtree_, dim_nums, mask, is_leaf=is_leaf)

  return combine.partition(
      transforms={
          'norprism': combine.chain(
              scale_by_norprism(
                  ns_coeffs=ns_coeffs,
                  ns_steps=ns_steps,
                  b1=b1,
                  b2=b2,
                  gamma=gamma,
                  eps=eps,
                  rms_scale=rms_scale,
                  mu_dtype=mu_dtype,
                  v_dtype=v_dtype,
                  nesterov=nesterov,
                  weight_dimension_numbers=norprism_weight_dim_nums_fn,
              ),
              transform.add_decayed_weights(weight_decay, weight_decay_mask),
              transform.scale_by_learning_rate(learning_rate),
          ),
          'adam': alias.adamw(
              learning_rate=adam_learning_rate,
              b1=adam_b1,
              b2=adam_b2,
              eps=eps,
              eps_root=adam_eps_root,
              weight_decay=adam_weight_decay,
              mu_dtype=mu_dtype,
              nesterov=nesterov,
          ),
      },
      param_labels=param_labels,
  )
