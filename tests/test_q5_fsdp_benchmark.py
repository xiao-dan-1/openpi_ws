"""Small two-CPU sharding checks; never load a model or execute local GPU training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from pipeline.q5.benchmark_fsdp import gpu_selectors


class FsdpProbeTests(unittest.TestCase):
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
