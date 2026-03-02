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
"""Prism optimizer.

Implementation of the Prism optimizer from
https://arxiv.org/pdf/2602.03096.

Prism uses an innovation-augmented momentum matrix before applying
Muon's Newton-Schulz polar decomposition.
"""

from typing import Any, Callable, Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from optax.contrib import _muon
from optax.transforms import _masking
import optax.tree

_DEFAULT_PRISM_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

_is_weight_dim_nums = lambda x: isinstance(x, _muon.MuonDimensionNumbers)


class PrismState(NamedTuple):
  """State for the Prism algorithm."""

  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  ns_coeffs: jax.typing.ArrayLike


def scale_by_prism(
    ns_coeffs: Union[
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
        tuple[
            tuple[
                jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike
            ],
            ...,
        ],
        str,
    ] = _DEFAULT_PRISM_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    gamma: jax.typing.ArrayLike = 1.0,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = True,
    preconditioning: Literal[
        'frobenius', 'spectral', 'aol', 'schatten'
    ] = 'schatten',
    weight_dimension_numbers: _muon.WeightDimNumOrFn | None = None,
) -> base.GradientTransformation:
  """Rescale updates according to the Prism algorithm.

  Prism computes an innovation-augmented momentum matrix
  ``[M_t; gamma * (G_t - M_t)]`` for each block and extracts the
  orthogonalized momentum rows from its polar factor.
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  if isinstance(ns_coeffs, str):
    if ns_coeffs != 'dion':
      raise ValueError(
          "Prism supports only 'dion' preset when ns_coeffs is a string"
      )
    ns_coeffs = _DEFAULT_PRISM_NS_COEFFS

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

    return PrismState(
        count=jnp.zeros([], jnp.int32), mu=mu, ns_coeffs=ns_coeffs_
    )

  def update_fn(updates, state, params=None):
    del params
    if callable(weight_dimension_numbers):
      resolved_weight_dim_nums = weight_dimension_numbers(updates)
    elif weight_dimension_numbers is None:
      resolved_weight_dim_nums = jax.tree.map(
          lambda x: _muon.MuonDimensionNumbers() if x.ndim == 2 else None,
          updates,
      )
    else:
      resolved_weight_dim_nums = weight_dimension_numbers

    mu = optax.tree.update_moment(updates, state.mu, beta, 1)

    # Prism uses innovation D_t = G_t - M_t where M_t is the raw EMA momentum.
    raw_momentum = mu
    if nesterov:
      momentum = jax.tree.map(
          lambda m, g: beta * m + (1 - beta) * g,
          raw_momentum,
          updates,
      )
    else:
      momentum = raw_momentum
    count_inc = numerics.safe_increment(state.count)

    def prism_direction(
        m_t: jax.Array,
        ema_m_t: jax.Array,
        g_t: jax.Array,
        dim_num: _muon.MuonDimensionNumbers | None,
    ) -> jax.Array:
      if dim_num is None:
        if m_t.ndim != 2:
          raise ValueError(
              'Prism requires rank-2 updates or explicit dimension numbers, '
              f'got shape={m_t.shape}'
          )
        dim_num = _muon.MuonDimensionNumbers()

      reshape_fn, inverse_fn = _muon._compute_muon_reshape(  # pylint: disable=protected-access
          m_t, dim_num
      )
      m_t_matrix = reshape_fn(m_t)
      innovation_matrix = reshape_fn(g_t - ema_m_t)
      augmented_matrix = jnp.concatenate(
          (m_t_matrix, gamma * innovation_matrix), axis=-2
      )
      orthogonalized_augmented = _muon.orthogonalize_via_newton_schulz(
          augmented_matrix,
          state.ns_coeffs,
          ns_steps,
          preconditioning,
          eps,
          _muon.MuonDimensionNumbers(reduction_axis=1, output_axis=2),
      )
      orthogonalized_momentum = orthogonalized_augmented[
          :, : m_t_matrix.shape[-2], :
      ]
      return inverse_fn(orthogonalized_momentum)

    updates = jax.tree.map(
        prism_direction,
        momentum,
        raw_momentum,
        updates,
        resolved_weight_dim_nums,
        is_leaf=_is_weight_dim_nums,
    )

    mu = optax.tree.cast(mu, mu_dtype)
    return updates, PrismState(
        count=count_inc,
        mu=mu,
        ns_coeffs=state.ns_coeffs,
    )

  return base.GradientTransformation(init_fn, update_fn)


def prism(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
        tuple[
            tuple[
                jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike
            ],
            ...,
        ],
        str,
    ] = _DEFAULT_PRISM_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    gamma: jax.typing.ArrayLike = 1.0,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: jax.typing.ArrayLike = 0.0,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = True,
    preconditioning: Literal[
        'frobenius', 'spectral', 'aol', 'schatten'
    ] = 'schatten',
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.95,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: jax.typing.ArrayLike = 0.0,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    muon_weight_dimension_numbers: _muon.WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = 0.2,
) -> base.GradientTransformation:
  """Prism optimizer with Adam fallback for non-matrix parameters.

  This composes :func:`scale_by_prism` with Muon shape scaling, weight decay,
  and learning-rate scaling for matrix blocks, and routes non-matrix parameters
  to AdamW.
  """

  if adam_learning_rate is None:
    adam_learning_rate = learning_rate

  # None at root indicates the default 2D rule.
  if muon_weight_dimension_numbers is None:
    param_labels = lambda params: jax.tree.map(
        lambda x: 'prism' if x.ndim == 2 else 'adam', params
    )
    muon_weight_dimension_numbers = _muon.MuonDimensionNumbers()
  else:

    def param_labels(params):
      dim_nums = (
          muon_weight_dimension_numbers(params)
          if callable(muon_weight_dimension_numbers)
          else muon_weight_dimension_numbers
      )
      populate_subtree_ = lambda dim_num, x: jax.tree.map(
          lambda y: 'prism' if dim_num is not None else 'adam', x
      )
      return jax.tree.map(
          populate_subtree_,
          dim_nums,
          params,
          is_leaf=lambda x: x is None or _is_weight_dim_nums(x),
      )

  def prism_weight_dim_nums_fn(params):
    dim_nums = (
        muon_weight_dimension_numbers(params)
        if callable(muon_weight_dimension_numbers)
        else muon_weight_dimension_numbers
    )
    mask = jax.tree.map(lambda label: label == 'prism', param_labels(params))
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
          'prism': combine.chain(
              scale_by_prism(
                  ns_coeffs=ns_coeffs,
                  ns_steps=ns_steps,
                  beta=beta,
                  gamma=gamma,
                  eps=eps,
                  mu_dtype=mu_dtype,
                  nesterov=nesterov,
                  preconditioning=preconditioning,
                  weight_dimension_numbers=prism_weight_dim_nums_fn,
              ),
              _muon.scale_by_shape(
                  weight_dimension_numbers=prism_weight_dim_nums_fn,
                  consistent_rms=consistent_rms,
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
