"""Report persistence checks with tiny observations and mocked model/simulator."""
import contextlib
import dataclasses
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

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from pipeline.common.reporting import artifact_directory, emit_report


class ReportingTests(unittest.TestCase):
    def test_temporary_work_files_are_cleaned_on_success_and_failure(self):
        for fail in [False, True]:
            with self.subTest(fail=fail):
                try:
                    with artifact_directory(None) as work:
                        (work / "worker.json").write_text("{}")
                        if fail:
                            raise RuntimeError("fixture failure")
                except RuntimeError:
                    pass
                self.assertFalse(work.exists())

    def test_explicit_artifacts_remain_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports"
            with self.assertRaises(RuntimeError), artifact_directory(output) as work:
                (work / "worker.log").write_text("failure details")
                raise RuntimeError("fixture failure")
            self.assertEqual((output / "worker.log").read_text(), "failure details")

    def test_report_is_always_printed_and_only_explicitly_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                emit_report({"status": "PASS"})
            self.assertEqual(json.loads(console.getvalue()), {"status": "PASS"})
            self.assertEqual(list(root.iterdir()), [])
            with contextlib.redirect_stdout(io.StringIO()):
                emit_report({"status": "PASS"}, root / "reports/result.json")
            self.assertEqual(json.loads((root / "reports/result.json").read_text()), {"status": "PASS"})


class InferenceReportTests(unittest.TestCase):
    def test_cli_prints_by_default_and_saves_only_when_requested(self):
        from pipeline.inference import check_policy
        for save in [False, True]:
            with self.subTest(save=save), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "reports/result.json"
                argv = ["check_policy", "--exp-name", "demo", "--step", "119"]
                if save:
                    argv += ["--output", str(output)]
                console = io.StringIO()
                client = mock.Mock()
                client.infer.return_value = {"actions": np.zeros((10, 7))}
                requests = [(0, i, {"observation/state": np.zeros(8)}) for i in range(3)]
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(console), \
                     mock.patch.object(check_policy, "WORKSPACE", root), \
                     mock.patch.object(check_policy, "prepare_client"), \
                     mock.patch.object(check_policy, "read_server_metadata", return_value={}), \
                     mock.patch.object(check_policy, "verify_server_metadata", return_value={"fixture": True}), \
                     mock.patch.object(check_policy, "sample_requests", return_value=requests), \
                     mock.patch("openpi_client.websocket_client_policy.WebsocketClientPolicy", return_value=client):
                    check_policy.main()
                text = console.getvalue()
                report = json.loads(text[text.index("{"):])
                self.assertEqual(report["samples_checked"], 3)
                if save:
                    self.assertEqual(json.loads(output.read_text()), report)
                else:
                    self.assertEqual(list(root.iterdir()), [])


@dataclasses.dataclass
class TinyModel:
    action_dim: int = 38


@dataclasses.dataclass
class TinyData:
    repo_id: str = "train"
    repo_root: str = "fixture"

    @property
    def repack_transforms(self):
        return SimpleNamespace(inputs=[])


class ValidationReportTests(unittest.TestCase):
    def test_default_can_repeat_with_old_report_and_explicit_output_is_saved(self):
        from pipeline.q5 import evaluate_validation as validation
        from pipeline.q5.runtime import Paths, ASSET_ID, sha256
        for save in [False, True]:
            with self.subTest(save=save), tempfile.TemporaryDirectory() as directory:
                paths = Paths.create(directory)
                checkpoint = paths.workspace / "checkpoints/pi05_q5/demo/119"
                stats = checkpoint / "assets" / ASSET_ID / "norm_stats.json"
                stats.parent.mkdir(parents=True)
                stats.write_text("{}")
                paths.split_manifest.parent.mkdir(parents=True)
                paths.split_manifest.write_text("{}")
                model = TinyModel()
                run = paths.workspace / "runs/pi05_q5/demo/config.json"
                run.parent.mkdir(parents=True)
                run.write_text(json.dumps(dict(profile="smoke", model=dataclasses.asdict(model),
                    split_manifest_sha256=sha256(paths.split_manifest), norm_stats_sha256=sha256(stats))))
                previous = paths.workspace / "checks/output/q5/validation/demo/119.json"
                previous.parent.mkdir(parents=True)
                previous.write_text("previous report")
                before = {p: p.read_bytes() for p in paths.workspace.rglob("*") if p.is_file()}
                output = paths.workspace / "explicit/result.json"
                argv = ["evaluate_validation", "--workspace", directory, "--gpu", "0",
                        "--exp-name", "demo", "--step", "119"]
                if save:
                    argv += ["--output", str(output)]
                config = SimpleNamespace(model=model, assets_dirs=paths.workspace,
                    data=SimpleNamespace(create=lambda *args: TinyData()))
                policy = SimpleNamespace(infer=lambda *args, **kwargs: {"actions": np.zeros((10, 38))})
                console = io.StringIO()
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(console), \
                     mock.patch.object(validation, "prepare_environment"), \
                     mock.patch.object(validation, "require_stats", return_value={"sha256": sha256(stats)}), \
                     mock.patch.object(validation, "validate_split", return_value={"validation": [
                         {"length": 2, "output_episode_index": 0, "source_bag": "ep002"}]}), \
                     mock.patch("jax.default_backend", return_value="gpu"), \
                     mock.patch("jax.device_count", return_value=1), \
                     mock.patch("pipeline.q5.config.build_config", return_value=config), \
                     mock.patch("openpi.policies.policy_config.create_trained_policy", return_value=policy), \
                     mock.patch("openpi.training.data_loader.create_torch_dataset", side_effect=lambda *args: [
                         {"actions": np.zeros((10, 38))} for _ in range(2)]):
                    validation.main()
                    if not save:
                        validation.main()
                self.assertEqual(previous.read_text(), "previous report")
                for p, content in before.items():
                    self.assertEqual(p.read_bytes(), content)
                if save:
                    report = json.loads(output.read_text())
                    self.assertEqual(report["status"], "PASS")
                    self.assertEqual(report["equal_episode_mae_rad"]["right_arm"], 0)
                else:
                    self.assertEqual({p for p in paths.workspace.rglob("*") if p.is_file()}, set(before))


class SimulationReportTests(unittest.TestCase):
    def test_camera_images_and_report_are_saved_only_with_output_directory(self):
        from pipeline.eval import check_simulation
        for save in [False, True]:
            with self.subTest(save=save), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "reports"
                argv = ["check_simulation"] + (["--output-dir", str(output)] if save else [])
                pixels = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
                observation = {"agentview_image": pixels, "robot0_eye_in_hand_image": pixels,
                    "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.zeros(4),
                    "robot0_gripper_qpos": np.zeros(2)}
                env = mock.Mock()
                env.step.return_value = (observation, 0, False, {})
                suite = SimpleNamespace(n_tasks=1, get_task=lambda i: object(), get_task_init_states=lambda i: [0])
                official = SimpleNamespace(benchmark=SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": lambda: suite}),
                    LIBERO_ENV_RESOLUTION=256, LIBERO_DUMMY_ACTION=np.zeros(7),
                    _get_libero_env=lambda *args: (env, "task"), _quat2axisangle=lambda q: np.zeros(3))
                spec = SimpleNamespace(name="_pipeline_libero_render_check", loader=SimpleNamespace(exec_module=lambda module: None))
                console = io.StringIO()
                with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules), \
                     contextlib.redirect_stdout(console), mock.patch.object(check_simulation, "WORKSPACE", root), \
                     mock.patch.object(check_simulation, "prepare_simulation"), \
                     mock.patch("importlib.util.spec_from_file_location", return_value=spec), \
                     mock.patch("importlib.util.module_from_spec", return_value=official):
                    check_simulation.main()
                report = json.loads(console.getvalue()[console.getvalue().index("{"):])
                self.assertEqual(report["status"], "PASS")
                env.close.assert_called_once()
                if save:
                    self.assertEqual({p.name for p in output.iterdir()}, {"result.json", "agentview.png", "wrist.png"})
                    self.assertEqual(json.loads((output / "result.json").read_text()), report)
                else:
                    self.assertEqual(list(root.iterdir()), [])
                    self.assertIsNone(report["cameras"]["agentview_image"]["image_path"])


class EvaluationArtifactTests(unittest.TestCase):
    def test_official_video_artifacts_are_temporary_unless_requested(self):
        from pipeline.eval import eval_libero
        for save in [False, True]:
            with self.subTest(save=save), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "reports"
                argv = ["eval_libero", "--exp-name", "demo", "--step", "119"]
                if save:
                    argv += ["--output-dir", str(output)]
                video_directories = []

                def evaluate(args):
                    videos = Path(args.video_out_path)
                    video_directories.append(videos)
                    videos.mkdir()
                    (videos / "fixture.mp4").write_bytes(b"mock video")
                    logging.info("Total episodes: 10")
                    logging.info("Total success rate: 0.0")

                official = SimpleNamespace(Args=SimpleNamespace, eval_libero=evaluate)
                spec = SimpleNamespace(name="_pipeline_libero_evaluator", loader=SimpleNamespace(exec_module=lambda module: None))
                console = io.StringIO()
                with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules), \
                     contextlib.redirect_stdout(console), mock.patch.object(eval_libero, "prepare_simulation"), \
                     mock.patch.object(eval_libero, "read_server_metadata", return_value={}), \
                     mock.patch.object(eval_libero, "verify_server_metadata", return_value={"fixture": True}), \
                     mock.patch("importlib.util.spec_from_file_location", return_value=spec), \
                     mock.patch("importlib.util.module_from_spec", return_value=official):
                    eval_libero.main()
                report = json.loads(console.getvalue())
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["episodes_completed"], 10)
                if save:
                    self.assertEqual(json.loads((output / "result.json").read_text()), report)
                    self.assertTrue((output / "videos/fixture.mp4").exists())
                else:
                    self.assertEqual(list(root.iterdir()), [])
                    self.assertTrue(all(not p.parent.exists() for p in video_directories))


class TransportArtifactTests(unittest.TestCase):
    def test_real_mock_server_keeps_artifacts_only_when_requested(self):
        from checks import check_policy_transport
        from pipeline.inference import check_policy, policy_rpc
        for save in [False, True]:
            with self.subTest(save=save), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "reports"
                argv = ["check_policy_transport"] + (["--output-dir", str(output)] if save else [])
                request = {"observation/image": np.zeros((224, 224, 3), np.uint8),
                    "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
                    "observation/state": np.arange(8, dtype=np.float32), "prompt": "fixture task"}
                requests = [(0, i, request) for i in range(3)]
                original_workspace = policy_rpc.WORKSPACE
                workspaces = []
                run = check_policy_transport.run_check

                def run_and_track(work):
                    workspaces.append(work)
                    return run(work)

                console = io.StringIO()
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(console), \
                     mock.patch.object(check_policy_transport, "WORKSPACE", root), \
                     mock.patch.object(check_policy_transport, "run_check", side_effect=run_and_track), \
                     mock.patch.object(check_policy, "sample_requests", return_value=requests):
                    check_policy_transport.main()
                self.assertEqual(policy_rpc.WORKSPACE, original_workspace)
                report = json.loads(console.getvalue())
                self.assertTrue(report["wrong_checkpoint_rejected"])
                self.assertEqual(report["real_observations_checked"], 3)
                if save:
                    self.assertEqual(json.loads((output / "result.json").read_text()), report)
                    self.assertTrue((output / "mock_server.log").is_file())
                else:
                    self.assertEqual(list(root.iterdir()), [])
                    self.assertTrue(all(not p.exists() for p in workspaces))


if __name__ == "__main__":
    unittest.main()
