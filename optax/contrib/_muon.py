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
"""Muon.

Implementation of the
[Muon optimizer](https://github.com/KellerJordan/modded-nanogpt)
by Keller Jordan
"""


import functools
import math
from collections import OrderedDict
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

  Muon defines an orthogonalization for 2D matrix weights for matrix-vector
  products:

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

_CacheKey = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
_ReshapePair = tuple[ReshapeFn, ReshapeFn]
_RESHAPE_CACHE: "OrderedDict[_CacheKey, _ReshapePair]" = OrderedDict()
_RESHAPE_CACHE_MAXSIZE = 128

_NS_UNROLL_THRESHOLD = 4
_VMAP_BATCH_SIZE_THRESHOLD = 32
_VMAP_ELEMENTS_THRESHOLD = 1 << 16


def _contains_tracer(tree: base.PyTree) -> bool:
  return any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(tree))


def _shape_signature(
    tree: base.PyTree,
) -> tuple[jax.tree_util.PyTreeDef, tuple]:
  shapes = tuple(getattr(x, 'shape', None) for x in jax.tree.leaves(tree))
  return jax.tree.structure(tree), shapes


def _shape_dtype_signature(
    tree: base.PyTree,
) -> tuple[
    jax.tree_util.PyTreeDef,
    tuple[tuple[tuple[int, ...] | None, Any], ...],
]:
  leaves = jax.tree.leaves(tree)
  shapes_and_dtypes = tuple(
      (getattr(x, 'shape', None), getattr(x, 'dtype', None)) for x in leaves
  )
  return jax.tree.structure(tree), shapes_and_dtypes


def _dim_nums_signature(
    tree: base.PyTree,
) -> tuple[jax.tree_util.PyTreeDef, tuple[Any, ...]]:
  def _leaf_signature(x):
    if isinstance(x, MuonDimensionNumbers):
      reduction = tuple(
          x.reduction_axis
          if isinstance(x.reduction_axis, (tuple, list))
          else (x.reduction_axis,)
      )
      output = tuple(
          x.output_axis
          if isinstance(x.output_axis, (tuple, list))
          else (x.output_axis,)
      )
      return (reduction, output)
    if isinstance(x, _masking.MaskedNode):
      return ('masked',)
    return None

  leaves = tuple(_leaf_signature(x) for x in jax.tree.leaves(tree))
  return jax.tree.structure(tree), leaves


def _require_static_int(value: Any, name: str) -> int:
  if isinstance(value, bool):
    raise TypeError(f'{name} must be a Python int, got bool.')
  try:
    return int(value)
  except (TypeError, ValueError) as err:
    raise TypeError(
        f'{name} must be a Python int (static) to allow unrolling.'
    ) from err


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
  cache_key = (x.shape, reduction_axes, output_axes)
  cached = _RESHAPE_CACHE.get(cache_key)
  if cached is not None:
    _RESHAPE_CACHE.move_to_end(cache_key)
    return cached
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
  _RESHAPE_CACHE[cache_key] = (reshape_fn, inverse_fn)
  if len(_RESHAPE_CACHE) > _RESHAPE_CACHE_MAXSIZE:
    _RESHAPE_CACHE.popitem(last=False)
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
    shape_cache_ignore_dtype: bool = False,
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
    shape_cache_ignore_dtype: Whether to ignore leaf dtypes in the shape cache
      key when reusing scaling factors.

  Returns:
    A `GradientTransformation` object.
  """
  scale_cache: dict[tuple[Any, tuple], base.Updates] = {}

  def _shape_scale(update: jax.Array, dim_nums: MuonDimensionNumbers):
    fan_in, fan_out = _get_shape_products(update, dim_nums)
    if consistent_rms is not None:
      return jnp.sqrt(jnp.maximum(fan_in, fan_out)) * consistent_rms
    return jnp.sqrt(jnp.maximum(1, fan_out / fan_in))

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

    scale_factors = None
    if (resolved_weight_dim_nums is not None
        and not callable(weight_dimension_numbers)
        and not _contains_tracer(updates)):
      update_signature = (
          _shape_signature(updates)
          if shape_cache_ignore_dtype
          else _shape_dtype_signature(updates)
      )
      cache_key = (
          update_signature,
          _dim_nums_signature(resolved_weight_dim_nums),
      )
      scale_factors = scale_cache.get(cache_key)
      if scale_factors is None:
        scale_factors = jax.tree.map(
            _shape_scale,
            updates,
            resolved_weight_dim_nums,
            is_leaf=_is_weight_dim_nums,
        )
        scale_cache[cache_key] = scale_factors

    if scale_factors is None:
      scaled_updates = jax.tree.map(
          scaling_fn,
          updates,
          resolved_weight_dim_nums,
          is_leaf=_is_weight_dim_nums,
      )
    else:
      scaled_updates = jax.tree.map(
          lambda g, s: s * g,
          updates,
          scale_factors,
      )
    return scaled_updates, state

  # Use the standard empty_state initializer, as this transform is stateless
  return base.GradientTransformation(base.init_empty_state, update_fn)


def _matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    precision: jax.lax.Precision | None,
) -> jax.Array:
  return jax.lax.dot_general(
      lhs,
      rhs,
      dimension_numbers=(((1,), (0,)), ((), ())),
      precision=precision,
  )


def _newton_schulz_iterator(
    x: jax.Array,
    coeffs: jax.Array,
    precision: jax.lax.Precision | None,
) -> jax.Array:
  # Implements Newton-Schulz step f(X) = c_0 X + c_1 (XX^T)X + c_2 (XX^T)^2X,
  # with quintic form f(X) = c_0 X + (c_1 A + c_2 AA)X, where A = XX^T.
  # The NS step has the property f(X) = f(X^T)^T. That is, we can get equivalent
  # result by transposing input and output. In particular, we may transpose X
  # when rows > cols for efficiency.
  a = _matmul(x, x.T.conj(), precision)
  b = coeffs[1] * a + coeffs[2] * _matmul(a, a, precision)
  return coeffs[0] * x + _matmul(b, x, precision)


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike = 5,
    eps: jax.typing.ArrayLike = 1e-8,
    dimension_numbers: MuonDimensionNumbers | None = None,
    matmul_precision: jax.lax.Precision | None = None,
    use_lax_map: bool | None = None,
    remat_orthogonalize: bool = False,
    ns_unroll_threshold: int = _NS_UNROLL_THRESHOLD,
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
    matmul_precision: Optional matmul precision for the Newton-Schulz step.
    use_lax_map: Whether to use `jax.lax.map` instead of `jax.vmap` for the
      batch axis to reduce peak memory. If `None`, uses a heuristic based on
      batch and matrix size.
    remat_orthogonalize: Whether to checkpoint the orthogonalization to reduce
      memory usage when using `jax.vmap`.
    ns_unroll_threshold: When `ns_coeffs` is 2D, unroll with `fori_loop` when
      the number of steps is small.

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
  ns_steps_ = None
  if ns_coeffs.ndim == 1:
    ns_steps_ = _require_static_int(ns_steps, 'ns_steps')

  def _orthogonalize(x):
    transposed = False
    if x.shape[0] > x.shape[1]:
      x = x.T
      transposed = True

    x /= jnp.linalg.norm(x) + eps  # Ensure spectral norm is at most 1
    ns_coeffs_ = ns_coeffs.astype(x.dtype)
    if ns_coeffs_.ndim == 1:
      x = jax.lax.fori_loop(
          0,
          ns_steps_,
          lambda _, x: _newton_schulz_iterator(x, ns_coeffs_, matmul_precision),
          x,
          unroll=True,
      )  # Unroll to ensure efficient composition with jax.vmap.
    else:
      steps = ns_coeffs_.shape[0]
      if steps <= ns_unroll_threshold:
        def _body(i, x):
          return _newton_schulz_iterator(x, ns_coeffs_[i], matmul_precision)
        x = jax.lax.fori_loop(0, steps, _body, x, unroll=True)
      else:
        x, _ = jax.lax.scan(
            lambda x, abc: (
                _newton_schulz_iterator(x, abc, matmul_precision),
                None,
            ),
            x,
            ns_coeffs_,
        )
    if transposed:
      x = x.T
    return x

  reshape_fn, inverse_fn = _compute_muon_reshape(x, dimension_numbers)
  orthogonalize_fn = _orthogonalize
  if remat_orthogonalize:
    orthogonalize_fn = jax.checkpoint(orthogonalize_fn)
  batched_x = reshape_fn(x)
  if use_lax_map is None:
    batch = batched_x.shape[0]
    m, n = batched_x.shape[1], batched_x.shape[2]
    use_lax_map = (
        batch >= _VMAP_BATCH_SIZE_THRESHOLD
        or batch * m * n >= _VMAP_ELEMENTS_THRESHOLD
    )
  if use_lax_map:
    return inverse_fn(jax.lax.map(orthogonalize_fn, batched_x))
  return inverse_fn(jax.vmap(orthogonalize_fn)(batched_x))


class MuonState(NamedTuple):
  """State for the Muon algorithm."""
  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  ns_coeffs: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.


class _MomentAndHat:

  __slots__ = ('mu', 'mu_hat')

  def __init__(self, mu: Any, mu_hat: Any):
    self.mu = mu
    self.mu_hat = mu_hat


def _inner_product(
    lhs: jax.Array,
    rhs: jax.Array,
    precision: jax.lax.Precision | None,
) -> jax.Array:
  if lhs.ndim == 0:
    return lhs.conj() * rhs
  dims = tuple(range(lhs.ndim))
  return jax.lax.dot_general(
      lhs.conj(),
      rhs,
      dimension_numbers=((dims, dims), ((), ())),
      precision=precision,
  )


def scale_by_muon(
    ns_coeffs: Union[
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
        tuple[tuple[jax.typing.ArrayLike, jax.typing.ArrayLike,
                    jax.typing.ArrayLike], ...],
    ] = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    matmul_precision: jax.lax.Precision | None = None,
    adaptive_precision: jax.lax.Precision | None = None,
    use_lax_map: bool | None = None,
    remat_orthogonalize: bool = False,
    ns_unroll_threshold: int = _NS_UNROLL_THRESHOLD,
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformation:
  r"""Rescale updates according to the Muon algorithm.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Args:
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to denominators to improve numerical stability.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    matmul_precision: Optional matmul precision for Newton-Schulz.
    adaptive_precision: Optional matmul precision for the adaptive inner
      product, allowing it to differ from orthogonalization precision.
    use_lax_map: Whether to use `jax.lax.map` for batch orthogonalization. If
      `None`, uses a heuristic based on batch and matrix size.
    remat_orthogonalize: Whether to checkpoint orthogonalization to reduce
      memory usage when using `jax.vmap`.
    ns_unroll_threshold: When `ns_coeffs` is 2D, unroll with `fori_loop` when
      the number of steps is small.
    weight_dimension_numbers: An optional tree with the same structure as the
      params of `MuonDimensionNumbers`s, specifying how to reshape the
      parameters before and after the orthogonalization OR a callable returning
      such a tree. None implies that all parameters are 2D matrices.

  Returns:
    A `GradientTransformation` object.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  ns_coeffs_arr = jnp.asarray(ns_coeffs)
  if ns_coeffs_arr.ndim > 2 or ns_coeffs_arr.shape[-1] != 3:
    raise ValueError(
        f'ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_arr.shape}'
    )
  ns_steps_ = None
  if ns_coeffs_arr.ndim == 1:
    ns_steps_ = _require_static_int(ns_steps, 'ns_steps')

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
    return MuonState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        ns_coeffs=ns_coeffs_arr,
    )

  def update_fn(updates, state, params=None):
    del params
    # TODO(rdyro): extend to _masking._mask_callable
    if callable(weight_dimension_numbers):
      # Populate weight_dim_nums if it's a callable. Use updates instead of
      # actual params since only shapes matter and params may not be provided.
      resolved_weight_dim_nums = weight_dimension_numbers(updates)
    else:
      resolved_weight_dim_nums = weight_dimension_numbers

    count_inc = numerics.safe_increment(state.count)
    mu_bias_correction = 1 - beta**numerics.safe_increment(count_inc)
    update_bias_correction = 1 - beta**count_inc

    def _moment_and_hat(g, m):
      if g is None or isinstance(g, _masking.MaskedNode):
        return _MomentAndHat(g, g)
      mu = (1 - beta) * g + beta * m
      if nesterov:
        mu_hat = (
            beta * (mu / mu_bias_correction.astype(mu.dtype))
            + (1 - beta) * (g / update_bias_correction.astype(g.dtype))
        )
      else:
        mu_hat = mu / update_bias_correction.astype(mu.dtype)
      return _MomentAndHat(mu, mu_hat)

    moment_and_hat = jax.tree.map(
        _moment_and_hat,
        updates,
        state.mu,
        is_leaf=lambda x: x is None or isinstance(x, _masking.MaskedNode),
    )
    mu = jax.tree.map(
        lambda x: x.mu,
        moment_and_hat,
        is_leaf=lambda x: isinstance(x, _MomentAndHat),
    )
    mu_hat = jax.tree.map(
        lambda x: x.mu_hat,
        moment_and_hat,
        is_leaf=lambda x: isinstance(x, _MomentAndHat),
    )
    # Apply Newton-schulz orthogonalization.
    updates = jax.tree.map(
        lambda x, dim_num: orthogonalize_via_newton_schulz(
            x,
            state.ns_coeffs,
            ns_steps_,
            eps,
            dim_num,
            matmul_precision,
            use_lax_map,
            remat_orthogonalize,
            ns_unroll_threshold,
        ),
        mu_hat, resolved_weight_dim_nums, is_leaf=_is_weight_dim_nums)
    if adaptive:
      # Scale the orthogonalized updates by the dual norm of the original
      # updates. See https://arxiv.org/abs/2409.20325 for the derivation.
      precision = adaptive_precision
      if precision is None:
        precision = matmul_precision
      updates = jax.tree.map(
          lambda x, y: _inner_product(x, y, precision) * y,
          mu_hat,
          updates,
      )

    mu = optax.tree.cast(mu, mu_dtype)
    return updates, MuonState(
        count=count_inc,
        mu=mu,
        ns_coeffs=state.ns_coeffs,
    )
  return base.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
        tuple[tuple[jax.typing.ArrayLike, jax.typing.ArrayLike,
                    jax.typing.ArrayLike], ...],
    ] = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: jax.typing.ArrayLike = 0.0,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    matmul_precision: jax.lax.Precision | None = None,
    adaptive_precision: jax.lax.Precision | None = None,
    use_lax_map: bool | None = None,
    remat_orthogonalize: bool = False,
    ns_unroll_threshold: int = _NS_UNROLL_THRESHOLD,
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: jax.typing.ArrayLike = 0.0,
    muon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    shape_cache_ignore_dtype: bool = False,
) -> base.GradientTransformation:
  r"""Muon: Momentum Orthogonalized by Newton-schulz.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Note that Muon is currently only defined for 2D parameters, i.e. matrices.
  This is because the Newton-Schulz iterator expects a matrix as input.
  The non-2D parameters are instead passed through an Adam optimizer.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to the denominator to improve numerical stability.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    matmul_precision: Optional matmul precision for Newton-Schulz.
    adaptive_precision: Optional matmul precision for the adaptive inner
      product, allowing it to differ from orthogonalization precision.
    use_lax_map: Whether to use `jax.lax.map` for batch orthogonalization. If
      `None`, uses a heuristic based on batch and matrix size.
    remat_orthogonalize: Whether to checkpoint orthogonalization to reduce
      memory usage when using `jax.vmap`.
    ns_unroll_threshold: When `ns_coeffs` is 2D, unroll with `fori_loop` when
      the number of steps is small.
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.
    muon_weight_dimension_numbers: An optional tree of `MuonDimensionNumbers`s,
      specifying how to reshape the parameters for orthogonalization otherwise
      muon parameters are assumed to be 2D matrices. A `None` value indicates
      that the parameter is not a muon parameter and will be optimized with
      Adam. A callable takes as input the params and returns a possibly masked
      pytree of specs, similar to `weight_decay_mask`. If not provided, muon is
      applied to all 2D parameters.
    consistent_rms: An optional float to activate consistent RMS scaling.
      Scales updates by `sqrt(max(fan_in, fan_out)) * consistent_rms` to make
      root mean square (RMS) shape-independent, like AdamW. `0.2` is recommended
      to match AdamW's empirical RMS. See <https://arxiv.org/abs/2502.16982>.
      If `None`, uses width scaling `sqrt(max(1, fan_out / fan_in))`.
    shape_cache_ignore_dtype: Whether to ignore leaf dtypes in the shape cache
      key when reusing scaling factors.

  Returns:
    The corresponding `GradientTransformation`.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024

    Liu et al., `Muon is Scalable for LLM Training`,
    <https://arxiv.org/abs/2502.16982>`_, 2025
  """
  # None at root indicates the default 2D rule.
  if muon_weight_dimension_numbers is None:
    param_labels = lambda params: jax.tree.map(
        lambda x: 'muon' if x.ndim == 2 else 'adam', params
    )
    muon_weight_dimension_numbers = MuonDimensionNumbers()
  else:
    def param_labels(params):
      dim_nums = (muon_weight_dimension_numbers(params)
                  if callable(muon_weight_dimension_numbers)
                  else muon_weight_dimension_numbers)
      populate_subtree_ = lambda dim_num, x: jax.tree.map(
          lambda y: 'muon' if dim_num is not None else 'adam', x)
      # Dimension numbers come first since they can be a prefix mask.
      return jax.tree.map(populate_subtree_, dim_nums, params,
                          is_leaf=lambda x: x is None or _is_weight_dim_nums(x))

  # We need to normalize the dimension numbers because they have to match the
  # tree structure of the masked muon state tree (see `combine.partition`).
  dim_nums_cache: dict[tuple[jax.tree_util.PyTreeDef, tuple], base.PyTree] = {}
  label_cache: dict[tuple[jax.tree_util.PyTreeDef, tuple], base.PyTree] = {}
  mask_cache: dict[tuple[jax.tree_util.PyTreeDef, tuple], base.PyTree] = {}

  def muon_weight_dim_nums_fn(params):
    # if muon_weight_dimension_numbers is None:
    #   return None
    # Normalize the dimension numbers for `combine.partition`.
    # Insert MaskedNode() where muon state will be masked out.
    cache_key = None
    if not _contains_tracer(params):
      cache_key = _shape_signature(params)
      cached = dim_nums_cache.get(cache_key)
      if cached is not None:
        return cached

    dim_nums = (muon_weight_dimension_numbers(params)
                if callable(muon_weight_dimension_numbers)
                else muon_weight_dimension_numbers)
    labels = None
    mask = None
    if cache_key is not None:
      labels = label_cache.get(cache_key)
      mask = mask_cache.get(cache_key)
    if labels is None:
      labels = param_labels(params)
      if cache_key is not None:
        label_cache[cache_key] = labels
    if mask is None:
      mask = jax.tree.map(lambda label: label == 'muon', labels)
      if cache_key is not None:
        mask_cache[cache_key] = mask
    is_leaf = lambda x: (x is None or _is_weight_dim_nums(x)
                         or isinstance(x, _masking.MaskedNode))
    populate_subtree_ = lambda dim_nums, submask: jax.tree.map(
        lambda m: dim_nums if m else _masking.MaskedNode(), submask)
    resolved = jax.tree.map(
        populate_subtree_, dim_nums, mask, is_leaf=is_leaf
    )
    if cache_key is not None:
      dim_nums_cache[cache_key] = resolved
    return resolved

  return combine.partition(
      transforms={
          'muon': combine.chain(
              scale_by_muon(
                  ns_coeffs=ns_coeffs,
                  ns_steps=ns_steps,
                  beta=beta,
                  eps=eps,
                  mu_dtype=mu_dtype,
                  nesterov=nesterov,
                  adaptive=adaptive,
                  matmul_precision=matmul_precision,
                  adaptive_precision=adaptive_precision,
                  use_lax_map=use_lax_map,
                  remat_orthogonalize=remat_orthogonalize,
                  ns_unroll_threshold=ns_unroll_threshold,
                  weight_dimension_numbers=muon_weight_dim_nums_fn,
              ),
              scale_by_shape(
                  weight_dimension_numbers=muon_weight_dim_nums_fn,
                  consistent_rms=consistent_rms,
                  shape_cache_ignore_dtype=shape_cache_ignore_dtype,
              ),
              transform.add_decayed_weights(weight_decay, weight_decay_mask),
              transform.scale_by_learning_rate(learning_rate),
          ),
          'adam': alias.adamw(
              learning_rate=learning_rate,
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
