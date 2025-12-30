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
"""Tests for the AdaGO optimizer in `_adago.py`."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from optax._src import test_utils
from optax.contrib import _adago
from optax.transforms import _masking


class AdaGOTest(absltest.TestCase):

  def test_v_sq_accumulation_clips_norm(self):
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    grads = {"w": jnp.ones((2, 2), dtype=jnp.float32)}

    tx = _adago.scale_by_adago(
        learning_rate=0.1,
        gamma=1.0,
        min_step_size=0.0,
        initial_accumulator_value=0.5,
    )
    state = tx.init(params)
    _, new_state = tx.update(grads, state, params)

    expected_v_sq = jnp.array(0.5**2 + 1.0**2, dtype=jnp.float32)
    test_utils.assert_trees_all_close(new_state.v_sq["w"], expected_v_sq)

  def test_partition_masks_non_matrix_params(self):
    params = {
        "w": jnp.ones((2, 3), dtype=jnp.float32),
        "b": jnp.ones((3,), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.ones_like, params)
    tx = _adago.adago(learning_rate=1e-3)
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)

    adago_state = state.inner_states["adago"].inner_state[0]
    adam_state = state.inner_states["adam"].inner_state
    self.assertIsInstance(adago_state.mu["b"], _masking.MaskedNode)
    self.assertIsInstance(adam_state[0].mu["w"], _masking.MaskedNode)
    test_utils.assert_tree_all_finite(updates)


if __name__ == "__main__":
  absltest.main()
