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
"""Tests for NorPrism optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from optax._src import test_utils
from optax.contrib import _muon
from optax.contrib import _norprism

UNSPECIFIED = object()


def get_updates(params, norprism_weight_dimension_numbers=UNSPECIFIED):
  if norprism_weight_dimension_numbers is UNSPECIFIED:
    opt = _norprism.norprism(learning_rate=1.0)
  else:
    opt = _norprism.norprism(
        learning_rate=1.0,
        norprism_weight_dimension_numbers=norprism_weight_dimension_numbers,
    )
  state = opt.init(params)
  grad = params  # assume loss = 1/2 * sum(params ** 2)
  updates, state = opt.update(grad, state, params=params)
  return updates, state


class NorPrismTest(parameterized.TestCase):

  def test_scale_by_norprism_sets_rms(self):
    key = jax.random.key(0)
    params = {'w': jax.random.normal(key, (8, 16))}
    tx = _norprism.scale_by_norprism(rms_scale=0.2)
    state = tx.init(params)
    updates, _ = tx.update(params, state)
    rms = jnp.sqrt(jnp.mean(jnp.square(updates['w'])))
    test_utils.assert_trees_all_close(
        rms, jnp.asarray(0.2, dtype=rms.dtype), rtol=1e-5, atol=1e-5
    )

  @parameterized.named_parameters(
      {
          'testcase_name': '3d_batch_axis',
          'input_shape': (2, 3, 4),
          'dim_nums': _muon.MuonDimensionNumbers(
              reduction_axis=0, output_axis=2
          ),
          'expected_v_shape': (3, 4),
      },
      {
          'testcase_name': '4d_multiple_output_axes',
          'input_shape': (2, 3, 4, 5),
          'dim_nums': _muon.MuonDimensionNumbers(
              reduction_axis=2, output_axis=(0, 3)
          ),
          'expected_v_shape': (3, 10),
      },
  )
  def test_v_state_shape(self, input_shape, dim_nums, expected_v_shape):
    params = {'w': jnp.ones(input_shape)}
    tx = _norprism.scale_by_norprism(weight_dimension_numbers={'w': dim_nums})
    state = tx.init(params)
    self.assertEqual(state.v['w'].shape, expected_v_shape)

  def test_scale_by_norprism_gamma_changes_direction(self):
    grads = {'w': jnp.arange(1, 17, dtype=jnp.float32).reshape((4, 4))}

    tx_gamma_0 = _norprism.scale_by_norprism(gamma=0.0)
    tx_gamma_1 = _norprism.scale_by_norprism(gamma=1.0)

    state_gamma_0 = tx_gamma_0.init(grads)
    state_gamma_1 = tx_gamma_1.init(grads)

    updates_gamma_0, _ = tx_gamma_0.update(grads, state_gamma_0, params=grads)
    updates_gamma_1, _ = tx_gamma_1.update(grads, state_gamma_1, params=grads)

    self.assertFalse(jnp.allclose(updates_gamma_0['w'], updates_gamma_1['w']))

  def test_scale_by_norprism_accepts_extra_ns_coeffs(self):
    grads = {'w': jnp.ones((2, 2), dtype=jnp.float32)}
    ns_coeffs = (
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),
    )
    tx = _norprism.scale_by_norprism(ns_coeffs=ns_coeffs, ns_steps=5)
    state = tx.init(grads)

    self.assertTrue(jnp.allclose(state.ns_coeffs, jnp.asarray(ns_coeffs[-5:])))

  def test_scale_by_norprism_raises_with_too_few_ns_coeffs(self):
    grads = {'w': jnp.ones((2, 2), dtype=jnp.float32)}
    ns_coeffs = (
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    )
    tx = _norprism.scale_by_norprism(ns_coeffs=ns_coeffs, ns_steps=5)

    with self.assertRaisesRegex(ValueError, 'Not enough coeffs'):
      tx.init(grads)

  def test_callable_weight_dim_nums(self):
    def weight_dim_nums_fn(params):
      fn_ = lambda x: _muon.MuonDimensionNumbers(0, 1) if x.ndim == 2 else None
      return jax.tree.map(fn_, params)

    opt = _norprism.norprism(
        learning_rate=1e-3,
        norprism_weight_dimension_numbers=weight_dim_nums_fn,
    )
    params = {'w': jnp.ones((10, 10)), 'b': jnp.ones((10,))}
    state = opt.init(params)
    _, _ = opt.update(params, state, params=params)

  def test_default_partitions_non_2d_to_adam(self):
    params = {'w': jnp.ones((10, 10)), 'b': jnp.ones((10,))}
    updates, _ = get_updates(params)
    self.assertEqual(updates['w'].shape, params['w'].shape)
    self.assertEqual(updates['b'].shape, params['b'].shape)

  def test_partition_with_tuple_params(self):
    params = (jnp.ones((3,)), jnp.ones((2, 2)))
    opt = _norprism.norprism(learning_rate=1e-3)
    state = opt.init(params)
    updates, _ = opt.update(params, state, params=params)
    self.assertEqual(jax.tree.structure(updates), jax.tree.structure(params))


if __name__ == '__main__':
  absltest.main()
