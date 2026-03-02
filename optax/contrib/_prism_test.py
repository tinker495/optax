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
"""Tests for Prism optimizer."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import optax

from optax.contrib import _muon
from optax.contrib import _prism


class PrismTest(absltest.TestCase):

  def test_prism_update_shape(self):
    params = {'w': jnp.ones((4, 4)), 'b': jnp.ones((4,))}
    opt = _prism.prism(learning_rate=1e-3)
    state = opt.init(params)
    updates, _ = opt.update(params, state, params)

    self.assertEqual(updates['w'].shape, params['w'].shape)
    self.assertEqual(updates['b'].shape, params['b'].shape)

  def test_scale_by_prism_has_no_second_moment_state(self):
    grads = {'w': jnp.arange(9, dtype=jnp.float32).reshape((3, 3))}
    tx = _prism.scale_by_prism()
    state = tx.init(grads)
    updates, state = tx.update(grads, state, params=grads)

    self.assertIsInstance(state, _prism.PrismState)
    self.assertFalse(hasattr(state, 'nu'))
    self.assertTrue(jnp.all(jnp.isfinite(updates['w'])))

  def test_scale_by_prism_uses_innovation_augmented_polar(self):
    grads = {'w': jnp.arange(1, 10, dtype=jnp.float32).reshape((3, 3))}
    tx = _prism.scale_by_prism(gamma=1.0)
    state = tx.init(grads)

    mu = optax.tree.update_moment(grads, state.mu, 0.95, 1)
    momentum = jax.tree.map(
        lambda m, g: 0.95 * m + 0.05 * g,
        mu,
        grads,
    )
    expected = jax.tree.map(
        lambda x: _muon.orthogonalize_via_newton_schulz(
            jnp.concatenate([x, grads['w'] - mu['w']], axis=0),
            state.ns_coeffs,
            5,
            'schatten',
            1e-8,
        )[: x.shape[0]],
        momentum,
    )

    updates, _ = tx.update(grads, state, params=grads)
    self.assertTrue(
        jnp.allclose(updates['w'], expected['w'], rtol=5e-4, atol=5e-4)
    )

  def test_scale_by_prism_gamma_zero_matches_orthogonalized_momentum(self):
    grads = {'w': jnp.arange(1, 10, dtype=jnp.float32).reshape((3, 3))}
    prism_tx = _prism.scale_by_prism(gamma=0.0)

    prism_state = prism_tx.init(grads)
    mu = optax.tree.update_moment(grads, prism_state.mu, 0.95, 1)
    momentum = jax.tree.map(lambda m, g: 0.95 * m + 0.05 * g, mu, grads)
    expected = _muon.orthogonalize_via_newton_schulz(
        momentum['w'],
        prism_state.ns_coeffs,
        5,
        'schatten',
        1e-8,
    )
    prism_updates, _ = prism_tx.update(grads, prism_state, params=grads)

    self.assertTrue(
        jnp.allclose(prism_updates['w'], expected, rtol=5e-4, atol=5e-4)
    )

  def test_scale_by_prism_accepts_extra_ns_coeffs(self):
    grads = {'w': jnp.ones((2, 2), dtype=jnp.float32)}
    ns_coeffs = (
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),
    )
    tx = _prism.scale_by_prism(ns_coeffs=ns_coeffs, ns_steps=5)
    state = tx.init(grads)

    self.assertTrue(jnp.allclose(state.ns_coeffs, jnp.asarray(ns_coeffs[-5:])))

  def test_scale_by_prism_raises_with_too_few_ns_coeffs(self):
    grads = {'w': jnp.ones((2, 2), dtype=jnp.float32)}
    ns_coeffs = (
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    )
    tx = _prism.scale_by_prism(ns_coeffs=ns_coeffs, ns_steps=5)

    with self.assertRaisesRegex(ValueError, 'Not enough coeffs'):
      tx.init(grads)

  def test_prism_uses_adam_for_non_matrix_params(self):
    params = {'w': jnp.ones((4, 4)), 'nested': {'b': jnp.ones((4,))}}
    grads = jax.tree.map(lambda x: x * 0.1, params)
    opt = _prism.prism(learning_rate=1e-3)
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)

    self.assertEqual(updates['nested']['b'].shape, params['nested']['b'].shape)
    self.assertTrue(jnp.all(jnp.isfinite(updates['nested']['b'])))

  def test_prism_uses_paper_default_adam_b2_for_non_matrix_params(self):
    params = {'b': jnp.ones((4,), dtype=jnp.float32)}
    grads = {'b': jnp.full((4,), 0.1, dtype=jnp.float32)}
    prism_tx = _prism.prism(learning_rate=1e-3)
    adam_tx = optax.adamw(
        learning_rate=1e-3,
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        eps_root=0.0,
        weight_decay=0.0,
        nesterov=True,
    )

    prism_state = prism_tx.init(params)
    adam_state = adam_tx.init(params)

    prism_updates, _ = prism_tx.update(grads, prism_state, params)
    adam_updates, _ = adam_tx.update(grads, adam_state, params)

    self.assertTrue(
        jnp.allclose(
            prism_updates['b'], adam_updates['b'], rtol=1e-6, atol=1e-6
        )
    )


if __name__ == '__main__':
  absltest.main()
