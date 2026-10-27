"""Pinned real TrainState regression on a tiny CPU model, not pi05 GPU capacity."""
import dataclasses
import importlib.util
import os
from pathlib import Path
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if any(importlib.util.find_spec(name) is None for name in ("jax", "flax", "optax", "openpi")):
    raise unittest.SkipTest("Requires the pinned OpenPI training environment")

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from openpi.training.utils import TrainState
from openpi.training import sharding
from openpi.shared import array_typing as at
from pipeline.train.benchmark_batch import synchronized_update


def make_case():
    graph, params = nnx.split(nnx.Dict(weight=nnx.Param(jnp.array([1.0, 2.0]))))
    tx = optax.adam(0.01)
    state = TrainState(step=jnp.array(0), params=params, model_def=graph,
                       opt_state=tx.init(params), tx=tx, ema_decay=None)

    @at.typecheck
    def update(rng, state: TrainState, target):
        del rng
        def loss(params):
            model = nnx.merge(state.model_def, params)
            return jnp.mean((model.weight.value - target) ** 2)
        value, grads = jax.value_and_grad(loss)(state.params)
        updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
        next_state = dataclasses.replace(state, step=state.step+1,
                    params=optax.apply_updates(state.params, updates), opt_state=opt_state)
        return next_state, {"loss": value, "grad_norm": optax.global_norm(grads)}

    mesh = sharding.make_mesh(1)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    step = jax.jit(update, in_shardings=(replicated, replicated, replicated),
                   out_shardings=(replicated, replicated), donate_argnums=(1,))
    return state, step, mesh, jax.random.key(42), jnp.zeros(2)


class TrainStateJitTest(unittest.TestCase):
    def test_reproduce_old_lowering_failure(self):
        state, step, mesh, rng, batch = make_case()
        with self.assertRaises(Exception) as raised:
            with sharding.set_mesh(mesh):
                step.lower(rng, state, batch).compile()
        self.assertIn("ArgInfo", str(raised.exception))

    def test_normal_jit_updates_real_trainstate(self):
        state, step, mesh, rng, batch = make_case()
        first_loss = None
        for _ in range(5):
            state, metrics = synchronized_update(step, mesh, rng, state, batch)
            first_loss = metrics["loss"] if first_loss is None else first_loss
            self.assertGreater(metrics["grad_norm"], 0)
        self.assertEqual(int(state.step), 5)
        self.assertLess(metrics["loss"], first_loss)
        self.assertLess(float(state.params["weight"].value[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
