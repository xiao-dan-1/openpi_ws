"""Unit tests for local and Docker LIBERO simulation entrypoints."""
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pipeline.eval import check_simulation, docker_runtime, eval_libero, runtime


class DockerRuntimeTests(unittest.TestCase):
    def test_build_docker_command_egl_and_output_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            ws = root / "ws"
            ws.mkdir()
            output_dir = root / "reports/eval_run_1"
            cmd = docker_runtime.build_docker_command(
                module="pipeline.eval.eval_libero",
                inner_args=["--host", "10.110.230.112", "--port", "8000", "--trials", "1"],
                workspace=ws,
                image="libero:latest",
                render_backend="egl",
                host="10.110.230.112",
                output_dir=output_dir,
            )
            self.assertEqual(cmd[:5], ["docker", "run", "--rm", "--network", "host"])
            self.assertIn("--gpus", cmd)
            self.assertIn("all", cmd)
            self.assertIn(f"{ws}:/workspace:ro", cmd)
            self.assertIn(f"{output_dir.parent}:/output_parent", cmd)
            self.assertIn("MUJOCO_GL=egl", cmd)
            self.assertIn("PYOPENGL_PLATFORM=egl", cmd)
            self.assertTrue(any(item.startswith("NO_PROXY=") and "10.110.230.112" in item for item in cmd))
            self.assertIn("libero:latest", cmd)
            shell_cmd = cmd[-1]
            self.assertIn("source /.venv/bin/activate && python -B -u -m pipeline.eval.eval_libero", shell_cmd)
            self.assertIn("--output-dir /output_parent/eval_run_1", shell_cmd)
            self.assertIn("chown -R", shell_cmd)
            self.assertNotIn("-u", cmd)
            self.assertTrue(output_dir.parent.is_dir())
            self.assertFalse(output_dir.exists())

    def test_build_docker_command_osmesa_omits_gpu_and_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cmd = docker_runtime.build_docker_command(
                module="pipeline.eval.check_simulation",
                inner_args=["--suite", "libero_spatial"],
                workspace=root,
                render_backend="osmesa",
            )
            self.assertNotIn("--gpus", cmd)
            self.assertIn("CUDA_VISIBLE_DEVICES=", cmd)
            self.assertIn("MUJOCO_GL=osmesa", cmd)

            existing = root / "already_exists"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                docker_runtime.build_docker_command(
                    module="pipeline.eval.eval_libero",
                    inner_args=[],
                    workspace=root,
                    output_dir=existing,
                )


class EvalEntrypointTests(unittest.TestCase):
    def test_docker_eval_libero_dispatches_without_exp_name_or_step(self):
        argv = [
            "eval_libero",
            "--docker",
            "--host", "10.110.230.112",
            "--port", "28126",
            "--suite", "libero_spatial",
            "--task-id", "0",
        ]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(runtime, "in_container", return_value=False), \
             mock.patch.object(runtime, "run_in_docker", return_value=0) as runner:
            eval_libero.main()
        runner.assert_called_once()
        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["module"], "pipeline.eval.eval_libero")
        self.assertEqual(kwargs["host"], "10.110.230.112")
        self.assertEqual(kwargs["render_backend"], "egl")
        self.assertEqual(kwargs["image"], "libero:latest")
        self.assertIn("--task-id", kwargs["inner_args"])
        self.assertIn("0", kwargs["inner_args"])
        trials_index = kwargs["inner_args"].index("--trials")
        self.assertEqual(kwargs["inner_args"][trials_index + 1], "50")

    def test_check_simulation_selects_docker_for_explicit_image(self):
        argv = ["check_simulation", "--image", "libero:custom", "--render-backend", "osmesa"]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(runtime, "in_container", return_value=False), \
             mock.patch.object(runtime, "run_in_docker", return_value=0) as runner:
            check_simulation.main()
        runner.assert_called_once()
        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["module"], "pipeline.eval.check_simulation")
        self.assertEqual(kwargs["image"], "libero:custom")
        self.assertEqual(kwargs["render_backend"], "osmesa")

    def test_check_simulation_uses_local_osmesa_by_default(self):
        class LocalSimulationStarted(Exception):
            pass

        with mock.patch.object(sys, "argv", ["check_simulation"]), \
             mock.patch.object(runtime, "in_container", return_value=False), \
             mock.patch.object(runtime, "run_in_docker") as docker, \
             mock.patch.object(runtime, "_ensure_local_runtime") as local, \
             mock.patch.object(check_simulation, "prepare_simulation", side_effect=LocalSimulationStarted) as prepare, \
             self.assertRaises(LocalSimulationStarted):
            check_simulation.main()
        docker.assert_not_called()
        self.assertEqual(local.call_args.args[1:],
                         ("pipeline.eval.check_simulation", check_simulation.WORKSPACE, "osmesa"))
        prepare.assert_called_once_with("osmesa")

    def test_legacy_exp_name_and_step_rejected(self):
        for bad in [["--exp-name", "demo", "--step", "119"], ["--port", "0"], ["--trials", "0"], ["--task-id", "-1"]]:
            with self.subTest(bad=bad), \
                 mock.patch.object(sys, "argv", ["eval_libero", *bad]), \
                 mock.patch.object(eval_libero, "dispatch_simulation") as dispatch, \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                eval_libero.main()
            dispatch.assert_not_called()

    def test_in_container_single_task_eval_records_server_metadata_without_runs_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite_obj = SimpleNamespace(
                n_tasks=10,
                get_task=lambda i: f"task_{i}",
                get_task_init_states=lambda i: [f"init_{i}"],
            )
            benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": lambda: suite_obj})
            seen = {}

            def evaluate(args):
                suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
                seen["n_tasks"] = suite.n_tasks
                seen["task"] = suite.get_task(0)
                seen["init"] = suite.get_task_init_states(0)
                logging.info("Total episodes: 1")
                logging.info("Total success rate: 1.0")

            official = SimpleNamespace(Args=SimpleNamespace, benchmark=benchmark, eval_libero=evaluate)
            spec = SimpleNamespace(name="_pipeline_libero_evaluator", loader=SimpleNamespace(exec_module=lambda m: None))
            console = io.StringIO()
            server_meta = {"pipeline": {"config": "pi05_libero", "experiment": "202611131535", "checkpoint_step": 29999}}
            argv = ["eval_libero", "--host", "10.110.230.112", "--port", "28126", "--task-id", "3", "--trials", "1"]
            with mock.patch.object(sys, "argv", argv), \
                 contextlib.redirect_stdout(console), \
                 mock.patch.object(eval_libero, "WORKSPACE", root), \
                 mock.patch.object(runtime, "in_container", return_value=True), \
                 mock.patch.object(runtime, "_ensure_local_runtime") as local, \
                 mock.patch.object(eval_libero, "prepare_simulation"), \
                 mock.patch.object(eval_libero, "read_server_metadata", return_value=server_meta), \
                 mock.patch("importlib.util.spec_from_file_location", return_value=spec), \
                 mock.patch("importlib.util.module_from_spec", return_value=official):
                eval_libero.main()
            local.assert_not_called()
            report = json.loads(console.getvalue())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["host"], "10.110.230.112")
            self.assertEqual(report["port"], 28126)
            self.assertEqual(report["task_id"], 3)
            self.assertEqual(report["trials_per_task"], 1)
            self.assertEqual(report["episodes_completed"], 1)
            self.assertEqual(report["task_success_rate"], 1.0)
            self.assertEqual(report["server_metadata"], server_meta)
            self.assertEqual(seen, {"n_tasks": 1, "task": "task_3", "init": ["init_3"]})

    def test_prepare_simulation_does_not_require_datasets_and_preserves_gpu_for_egl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libero_base = root / "openpi/third_party/libero/libero/libero"
            for sub in ("bddl_files", "init_files", "assets"):
                (libero_base / sub).mkdir(parents=True)
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False), \
                 mock.patch.object(eval_libero, "WORKSPACE", root), \
                 mock.patch.object(sys, "version_info", (3, 8, 20)), \
                 mock.patch.object(sys, "path", sys.path.copy()):
                eval_libero.prepare_simulation("egl")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
                self.assertEqual(os.environ["MUJOCO_GL"], "egl")
                config = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
                self.assertEqual(json.loads(config.read_text())["benchmark_root"], str(libero_base))
                eval_libero.prepare_simulation("osmesa")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")

    def test_local_launcher_uses_workspace_python_and_osmesa_library(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            python = workspace / "tools/libero-env/bin/python"
            python.parent.mkdir(parents=True)
            python.touch(mode=0o755)
            library_dir = workspace / "tools/osmesa/usr/lib/x86_64-linux-gnu"
            library_dir.mkdir(parents=True)
            (library_dir / "libOSMesa.so.8").touch()
            module = "pipeline.eval.eval_libero"
            argv = ["eval_libero", "--host", "127.0.0.1", "--task-id", "3"]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(sys, "version_info", (3, 11, 0)), \
                 mock.patch.object(runtime.sysconfig, "get_config_var", return_value="x86_64-linux-gnu"), \
                 mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/custom/lib"}), \
                 mock.patch.object(os, "execve") as exec_python:
                runtime._ensure_local_runtime(None, module, workspace, "osmesa")
            executable, command, env = exec_python.call_args.args
            self.assertEqual(executable, python)
            self.assertEqual(command, [str(python), "-B", "-u", "-m", module, *argv[1:]])
            self.assertEqual(env["LD_LIBRARY_PATH"].split(os.pathsep), ["/custom/lib", str(library_dir)])


if __name__ == "__main__":
    unittest.main()
