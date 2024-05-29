# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Common tests for contributed optimizers.

Additional specific tests are implemented in additional files
(see e.g. sam_test)
"""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from functools import partial
import jax
import jax.numpy as jnp
from optax import contrib
from optax._src import numerics
from optax._src import update
from optax.schedules import _inject
from optax.tree_utils import _state_utils

# Testing contributions coded as GradientTransformations
_OPTIMIZERS_UNDER_TEST = (
    dict(opt_name='acprop', opt_kwargs=dict(learning_rate=1e-3)),
    dict(opt_name='cocob', opt_kwargs=dict(alpha=100.0, eps=1e-8)),
    dict(opt_name='cocob', opt_kwargs=dict(weight_decay=1e-2)),
    dict(opt_name='dadapt_adamw', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='dog', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='dowg', opt_kwargs=dict(learning_rate=1.0)),
    dict(opt_name='momo', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='momo_adam', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='prodigy', opt_kwargs=dict(learning_rate=1e-1)),
    dict(opt_name='sophia_h', opt_kwargs=dict(learning_rate=1e-2)),
)


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  def loss_fn(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  get_updates = jax.value_and_grad(loss_fn)

  return initial_params, final_params, get_updates, loss_fn


def _setup_rosenbrock(dtype):
  """Rosenbrock function as an optimization target."""
  a = 1.0
  b = 100.0

  initial_params = jnp.array([0.0, 0.0], dtype=dtype)
  final_params = jnp.array([a, a**2], dtype=dtype)

  def loss_fn(params):
    return numerics.abs_sq(a - params[0]) + b * numerics.abs_sq(
        params[1] - params[0] ** 2
    )

  get_updates = jax.value_and_grad(loss_fn)

  return initial_params, final_params, get_updates, loss_fn


class ContribTest(chex.TestCase):

  @parameterized.product(
      _OPTIMIZERS_UNDER_TEST,
      target=(_setup_parabola, _setup_rosenbrock),
      dtype=(jnp.float32,),
  )
  def test_optimizers(self, opt_name, opt_kwargs, target, dtype):
    opt = getattr(contrib, opt_name)(**opt_kwargs)
    initial_params, final_params, get_updates, loss_fn = target(dtype)

    @jax.jit
    def step(params, state):
      value, updates = get_updates(params)
      if opt_name in ['momo', 'momo_adam']:
        update_kwargs = {'value': value}
      elif opt_name == 'sophia_h':
        update_kwargs = {'obj_fn': loss_fn}
      else:
        update_kwargs = {}
      updates, state = opt.update(updates, state, params, **update_kwargs)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    # A no-op change, to verify that tree map works.
    state = _state_utils.tree_map_params(opt, lambda v: v, state)

    def f(params_state, _):
      return step(*params_state), None

    (params, _), _ = jax.lax.scan(f, (params, state), length=30_000)

    chex.assert_trees_all_close(params, final_params, rtol=3e-2, atol=3e-2)

  @chex.all_variants
  @parameterized.product(_OPTIMIZERS_UNDER_TEST)
  def test_optimizers_can_be_wrapped_in_inject_hyperparams(
      self, opt_name, opt_kwargs
  ):
    """Checks that optimizers can be wrapped in inject_hyperparams."""
    # See also https://github.com/deepmind/optax/issues/412.
    opt_factory = getattr(contrib, opt_name)
    opt = opt_factory(**opt_kwargs)
    opt_inject = _inject.inject_hyperparams(opt_factory)(**opt_kwargs)

    params = [jnp.negative(jnp.ones((2, 3))), jnp.ones((2, 5, 2))]
    grads = [jnp.ones((2, 3)), jnp.negative(jnp.ones((2, 5, 2)))]

    opt_update = opt.update
    opt_inject_update = opt_inject.update
    if opt_name in ['momo', 'momo_adam']:
      update_kwargs = {'value': jnp.array(1.0)}
    elif opt_name == 'sophia_h':
      temp_update_kwargs = {
          'obj_fn': lambda ps: sum(jnp.sum(p) for p in jax.tree.leaves(ps))
      }
      opt_update = partial(opt.update, **temp_update_kwargs)
      opt_inject_update = partial(opt_inject.update, **temp_update_kwargs)
      update_kwargs = {}
    else:
      update_kwargs = {}

    state = self.variant(opt.init)(params)
    updates, new_state = self.variant(opt_update)(
        grads, state, params, **update_kwargs
    )

    state_inject = self.variant(opt_inject.init)(params)
    updates_inject, new_state_inject = self.variant(opt_inject_update)(
        grads, state_inject, params, **update_kwargs
    )

    with self.subTest('Equality of updates.'):
      chex.assert_trees_all_close(updates_inject, updates, rtol=1e-4)
    with self.subTest('Equality of new optimizer states.'):
      chex.assert_trees_all_close(
          new_state_inject.inner_state, new_state, rtol=1e-4
      )


if __name__ == '__main__':
  absltest.main()
