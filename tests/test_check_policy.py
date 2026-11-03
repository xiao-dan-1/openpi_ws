"""Q5 client routing, policy identity and actual video/state request alignment."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from pipeline.inference import check_policy, policy_rpc
from pipeline.q5.schema import JOINT_NAMES, ACTION_REPRESENTATION


class CheckPolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        record = {"model": {"action_dim": 38}, "norm_stats_sha256": "training-statistics"}
        path = self.root / "runs/pi05_q5/demo/config.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record))
        self.metadata = {"robot_type": "q5", "joint_names": list(JOINT_NAMES),
                         "action_representation": ACTION_REPRESENTATION,
                         "pipeline": {"experiment": "demo", "checkpoint_step": 15999,
                                      "upstream_commit": policy_rpc.UPSTREAM_COMMIT, **record}}

    def test_q5_identity_rejects_wrong_checkpoint_statistics_model_and_joint_contract(self):
        self.assertEqual(policy_rpc.verify_server_metadata(self.metadata, "demo", 15999,
                         workspace=self.root, robot="q5"), self.metadata["pipeline"])
        for field in ["checkpoint_step", "norm_stats_sha256", "model", "robot_type", "joint_names", "action_representation"]:
            changed = copy.deepcopy(self.metadata)
            target = changed["pipeline"] if field in changed["pipeline"] else changed
            target[field] = "changed"
            with self.subTest(field=field), self.assertRaises(ValueError):
                policy_rpc.verify_server_metadata(changed, "demo", 15999, workspace=self.root, robot="q5")

    def test_legacy_identity_defaults_to_libero(self):
        path = self.root / "runs/pi05_libero/demo/config.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"model": self.metadata["pipeline"]["model"],
                                    "norm_stats_sha256": "training-statistics"}))
        metadata = {"pipeline": self.metadata["pipeline"]}
        self.assertEqual(policy_rpc.verify_server_metadata(metadata, "demo", 15999, workspace=self.root),
                         metadata["pipeline"])

    def invoke(self, actions, *extra):
        argv = ["check_policy", "--robot", "q5", "--exp-name", "demo", "--step", "15999", *extra]
        client = mock.Mock()
        client.get_server_metadata.return_value = self.metadata
        client.infer.return_value = {"actions": actions}
        request = {"state": np.zeros(38), "images": {"image_head": np.zeros((8, 8, 3), np.uint8)}, "prompt": "pick"}
        console = io.StringIO()
        with mock.patch.dict(os.environ), mock.patch.object(sys, "argv", argv), \
             mock.patch.object(check_policy, "WORKSPACE", self.root), contextlib.redirect_stdout(console), \
             mock.patch.object(check_policy, "read_server_metadata", return_value=self.metadata), \
             mock.patch.object(check_policy, "sample_requests") as libero, \
             mock.patch.object(check_policy, "q5_sample_requests", return_value=[(0, 0, request)]) as q5, \
             mock.patch("openpi_client.websocket_client_policy.WebsocketClientPolicy", return_value=client):
            check_policy.main()
        libero.assert_not_called()
        return client, q5, json.loads(console.getvalue()[console.getvalue().index("{"):])

    def test_q5_cli_defaults_and_explicit_episodes_use_38d_actions(self):
        for extra, selected in [((), [0, 2, 4]), (("--episodes", "1", "3"), [1, 3])]:
            with self.subTest(selected=selected):
                client, sampler, report = self.invoke(np.zeros((10, 38)), *extra)
                sampler.assert_called_once_with(selected)
                client.infer.assert_called_once()
                self.assertEqual(report["robot"], "q5")
                self.assertEqual(report["samples"][0]["actions_shape"], [10, 38])
                self.assertFalse((self.root / "reports").exists())

    def test_wrong_response_shape_and_nan_are_rejected(self):
        for actions in [np.zeros((10, 7)), np.full((10, 38), np.nan)]:
            with self.subTest(shape=actions.shape), self.assertRaisesRegex(ValueError, "Invalid policy actions"):
                self.invoke(actions)


class Q5ObservationTests(unittest.TestCase):
    def setUp(self):
        import av
        import pyarrow as pa
        import pyarrow.parquet as pq

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "datasets/q5/pick_orange_to_basket_v2_validation"
        (self.data / "meta").mkdir(parents=True)
        self.info = {"total_episodes": 1, "chunks_size": 1000, "fps": 30,
                     "data_path": "episode_{episode_index:06d}.parquet",
                     "video_path": "episode_{episode_index:06d}.mkv",
                     "features": {"observation.state": {"names": list(JOINT_NAMES)}}}
        self.info_path = self.data / "meta/info.json"
        self.info_path.write_text(json.dumps(self.info))
        (self.data / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "放橙子"}) + "\n")
        self.rows = [{"observation.state": [float(frame)] * 38, "task_index": 0, "timestamp": frame / 30}
                     for frame in range(5)]
        self.parquet = self.data / "episode_000000.parquet"
        pq.write_table(pa.Table.from_pylist(self.rows), self.parquet)
        with av.open(str(self.data / "episode_000000.mkv"), "w") as output:
            stream = output.add_stream("ffv1", rate=30)
            stream.width, stream.height, stream.pix_fmt = 16, 16, "bgr0"
            for frame in range(5):
                image = av.VideoFrame.from_ndarray(np.full((16, 16, 3), frame * 30, np.uint8), format="rgb24")
                for packet in stream.encode(image):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)

    def requests(self, episodes=None):
        with mock.patch.object(check_policy, "WORKSPACE", self.root):
            return list(check_policy.q5_sample_requests([0] if episodes is None else episodes))

    def test_first_middle_last_frames_match_state_and_actual_video_pixels(self):
        requests = self.requests()
        self.assertEqual([frame for _, frame, _ in requests], [0, 2, 4])
        for _, frame, request in requests:
            np.testing.assert_array_equal(request["state"], np.full(38, frame, np.float32))
            np.testing.assert_array_equal(request["images"]["image_head"], np.full((16, 16, 3), frame * 30, np.uint8))
            self.assertEqual(request["prompt"], "放橙子")
            self.assertNotIn("actions", request)
        self.assertFalse((self.root / "datasets/q5/pick_orange_to_basket_v2_test").exists())

    def test_bad_timestamps_state_joint_order_or_episode_are_rejected(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        for field in ["timestamp", "observation.state"]:
            changed = copy.deepcopy(self.rows)
            changed[0][field] = 1.0 if field == "timestamp" else [float("nan")] * 38
            pq.write_table(pa.Table.from_pylist(changed), self.parquet)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "timestamp|state"):
                self.requests()
        pq.write_table(pa.Table.from_pylist(self.rows), self.parquet)
        with self.assertRaisesRegex(ValueError, "Invalid Q5 validation episode"):
            self.requests([1])
        self.info["features"]["observation.state"]["names"] = ["wrong"]
        self.info_path.write_text(json.dumps(self.info))
        with self.assertRaisesRegex(ValueError, "joint order"):
            self.requests()

    def test_video_row_count_mismatch_rejected(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(self.rows[:-1]), self.parquet)
        with self.assertRaisesRegex(ValueError, "video/frame count differs"):
            self.requests()


if __name__ == "__main__":
    unittest.main()
