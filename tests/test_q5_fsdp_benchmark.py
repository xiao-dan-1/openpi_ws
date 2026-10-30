"""Small two-CPU sharding checks; never load a model or execute local GPU training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from pipeline.q5.benchmark_fsdp import checked_metrics, gpu_selectors, measure_throughput, parse_args


class FsdpProbeTests(unittest.TestCase):
    def test_worker_and_timing_options_preserve_the_baseline(self):
        arguments = ["--gpus", "0", "--fsdp-devices", "1"]
        baseline = parse_args(arguments)
        self.assertEqual(baseline.num_workers, 0)
        self.assertEqual(baseline.timing_mode, "diagnostic")
        configured = parse_args(arguments + ["--num-workers", "4", "--timing-mode", "throughput"])
        self.assertEqual(configured.num_workers, 4)
        self.assertEqual(configured.timing_mode, "throughput")

    def test_throughput_waits_at_both_boundaries_and_keeps_metric_reads_outside_timing(self):
        import jax
        import jax.numpy as jnp
        events = []
        original_wait = jax.block_until_ready
        original_checked = checked_metrics

        def wait(value):
            events.append("wait")
            return original_wait(value)

        def clock():
            events.append("clock")
            return 10.0 if events.count("clock") == 1 else 12.0

        @jax.jit
        def compiled(state, batch):
            return state + batch, {"loss": state * 0.5, "grad_norm": jnp.ones(())}

        def update(state, batch):
            events.append("update")
            return compiled(state, batch)

        def checked(info):
            events.append("read_metrics")
            return original_checked(info)

        with mock.patch("jax.block_until_ready", side_effect=wait), \
             mock.patch("pipeline.q5.benchmark_fsdp.time.perf_counter", side_effect=clock), \
             mock.patch("pipeline.q5.benchmark_fsdp.checked_metrics", side_effect=checked):
            state, metrics, seconds = measure_throughput(
                jnp.array(0.0), iter([jnp.array(2.0), jnp.array(3.0)]), update, 2)
        self.assertEqual(float(state), 5.0)
        self.assertEqual(seconds, 2.0)
        self.assertEqual([row["loss"] for row in metrics], [0.0, 1.0])
        self.assertEqual(events, ["wait", "clock", "update", "update", "wait", "clock",
                                  "read_metrics", "read_metrics"])

    def test_nonfinite_metrics_fail_instead_of_reporting_throughput_success(self):
        import jax.numpy as jnp
        for invalid in [float("nan"), float("inf")]:
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Non-finite"):
                measure_throughput(jnp.array(0), iter([None, None]),
                                   lambda state, batch: (state + 1, {"loss": jnp.array(invalid)}), 2)

    def test_gpu_selection_is_explicit_and_bounded(self):
        self.assertEqual(gpu_selectors("2,3"), "2,3")
        self.assertEqual(gpu_selectors("2"), "2")
        for value in ["", "2,2", "0,1,2", "-1", "0,x"]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                gpu_selectors(value)

    def test_normal_training_environment_stays_single_gpu(self):
        from pipeline.q5 import runtime
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ), \
             mock.patch.object(runtime.subprocess, "check_output", side_effect=[runtime.UPSTREAM_COMMIT, ""]), \
             mock.patch.object(runtime, "configure_video_backend"):
            with self.assertRaisesRegex(ValueError, "one server GPU"):
                runtime.prepare_environment(runtime.Paths.create(temporary), cpu=False, gpu="2,3")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ), \
             mock.patch.object(runtime.subprocess, "check_output", side_effect=[runtime.UPSTREAM_COMMIT, ""]), \
             mock.patch.object(runtime, "configure_video_backend"):
            runtime.prepare_environment(runtime.Paths.create(temporary), cpu=False, gpu="2,3", allow_multiple_gpus=True)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3")
            self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.75")

    def test_two_cpu_devices_really_partition_state_and_execute_donated_update(self):
        code = '''
import jax, numpy as np
from types import SimpleNamespace
from openpi.training import sharding
from pipeline.q5.benchmark_fsdp import sharded_state_storage
from pipeline.q5.check_model import compile_training_step
assert jax.default_backend() == 'cpu' and jax.device_count() == 2
mesh = sharding.make_mesh(2)
assert mesh.shape['fsdp'] == 2 and mesh.shape['batch'] == 1
shape = {'params': jax.ShapeDtypeStruct((2048,1024), np.float32)}
layout = sharding.fsdp_sharding(shape, mesh)
state = {'params': jax.device_put(np.ones((2048,1024),np.float32), layout['params'])}
storage = sharded_state_storage(state)
assert storage['sharded_array_leaves'] == 1
assert len(storage['per_device_bytes']) == 2
assert all(n == storage['logical_bytes']//2 for n in storage['per_device_bytes'].values())
batch_layout = jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
batch = jax.device_put(np.ones((4,38),np.float32),batch_layout)
assert [s.data.shape for s in batch.addressable_shards] == [(2,38),(2,38)]
def toy_step(config,rng,current,b):
    return {'params':current['params']-b.mean()*0.1}, {'loss':current['params'].mean()}
step=compile_training_step(SimpleNamespace(train_step=toy_step),None,mesh,layout)
old=state['params']
with sharding.set_mesh(mesh):
    state,info=step(jax.random.key(42),state,batch)
jax.block_until_ready(state)
assert old.is_deleted()
assert abs(float(info['loss'])-1)<1e-6
assert abs(float(state['params'].mean())-0.9)<1e-4
print('PASS two-device FSDP state layout, split global batch and donated update')
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                   XLA_FLAGS="--xla_force_host_platform_device_count=2")
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS two-device", result.stdout)


if __name__ == "__main__":
    unittest.main()
