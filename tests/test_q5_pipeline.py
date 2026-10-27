"""Small CPU checks; never initialize the full model, load weights, or train."""
import copy
import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.traverse_util
import jax
import numpy as np

from pipeline.q5.data import Q5Inputs, Q5Outputs
from pipeline.q5.prepare_data import choose_test_ids, choose_validation_ids
from pipeline.q5.schema import JOINT_NAMES
from pipeline.q5.weight_loaders import merge_trunk, PROJECTION_KEYS, Q5WeightLoader
from pipeline.q5.runtime import Paths, require_stats, sha256, validate_split, configure_video_backend


class Q5ContractTests(unittest.TestCase):
    def test_validation_comes_only_from_previous_training_candidates(self):
        tests = set(choose_test_ids(42))
        validation = set(choose_validation_ids(tests))
        self.assertEqual(len(validation), 5)
        self.assertTrue(tests.isdisjoint(validation))
        self.assertEqual(len(set(range(65)) - tests - validation), 50)
        self.assertEqual(validation, set(choose_validation_ids(tests)))

    def test_pyav_replaces_both_automatic_selectors_without_codec_detection(self):
        from lerobot.common.datasets import lerobot_dataset, video_utils
        def broken_default():
            raise RuntimeError("TorchCodec native libraries are unavailable")
        with mock.patch.object(lerobot_dataset, "get_safe_default_codec", broken_default), \
             mock.patch.object(video_utils, "get_safe_default_codec", broken_default), \
             mock.patch("importlib.util.find_spec", side_effect=AssertionError("No codec autodetection allowed")):
            configure_video_backend()
            self.assertEqual(lerobot_dataset.get_safe_default_codec(), "pyav")
            self.assertEqual(video_utils.get_safe_default_codec(), "pyav")

    def test_split_is_repeatable_disjoint_and_covers_collection(self):
        selected = choose_test_ids(42)
        self.assertEqual(selected, choose_test_ids(42))
        self.assertNotEqual(selected, choose_test_ids(43))
        self.assertEqual(len(set(selected)), 10)
        self.assertEqual(len(set(range(65)) - set(selected)), 55)
        for i in range(10):
            self.assertEqual(sum(i * 65 // 10 <= n < (i + 1) * 65 // 10 for n in selected), 1)

    def test_contract_keeps_right_pinky_as_last_joint(self):
        self.assertEqual(len(set(JOINT_NAMES)), 38)
        self.assertEqual(JOINT_NAMES[14], "left_hand_thumb_bend_joint")
        self.assertEqual(JOINT_NAMES[37], "right_hand_pinky_joint2")

    def test_input_output_preserve_every_joint_and_camera_mask(self):
        state = np.arange(38, dtype=np.float32)
        actions = np.tile(state, (10, 1))
        result = Q5Inputs()({"state": state, "actions": actions,
                             "images": {"image_head": np.ones((3, 8, 12), dtype=np.float32)}, "prompt": "task"})
        np.testing.assert_array_equal(result["state"], state)
        np.testing.assert_array_equal(Q5Outputs()(result)["actions"], actions)
        self.assertEqual(result["image"]["base_0_rgb"].shape, (8, 12, 3))
        self.assertEqual(result["image"]["base_0_rgb"].dtype, np.uint8)
        self.assertEqual(result["image_mask"], {"base_0_rgb": True, "left_wrist_0_rgb": False, "right_wrist_0_rgb": False})

    def test_invalid_or_short_output_is_rejected(self):
        for actions in [np.zeros((10, 32)), np.full((10, 38), np.nan), np.zeros(38)]:
            with self.assertRaises(ValueError):
                Q5Outputs()({"actions": actions})


class Q5BaselineParityTests(unittest.TestCase):
    def test_freeze_rule_matches_libero_including_vision_and_time_mlp(self):
        import flax.nnx as nnx
        from pipeline.common.config import build_config as libero_config
        from pipeline.q5.config import build_config
        q5 = build_config(Paths.create(tempfile.gettempdir()))
        libero = libero_config()
        actual = nnx.filterlib.to_predicate(q5.trainable_filter)
        expected = nnx.filterlib.to_predicate(libero.trainable_filter)
        variable = nnx.Param(np.zeros(1, dtype=np.float32))
        cases = [(('PaliGemma', 'img', 'kernel'), True), (('time_mlp_in', 'kernel'), True),
                 (('time_mlp_out', 'kernel'), True), (('action_in_proj', 'kernel'), True),
                 (('action_out_proj', 'bias'), True), (('PaliGemma', 'llm', 'kernel'), False),
                 (('PaliGemma', 'llm', 'test_lora', 'a'), True)]
        for path, trainable in cases:
            with self.subTest(path=path):
                self.assertEqual(actual(path, variable), expected(path, variable))
                self.assertEqual(actual(path, variable), trainable)

    def test_schedule_and_retention_match_libero_without_committing_to_long_run(self):
        from pipeline.common.config import build_config as libero_config
        from pipeline.q5.config import build_config
        for profile in ["smoke", "finetune"]:
            with self.subTest(profile=profile):
                q5 = build_config(Paths.create(tempfile.gettempdir()), profile=profile)
                libero = libero_config(profile=profile)
                self.assertEqual(q5.lr_schedule, libero.lr_schedule)
                self.assertEqual(q5.optimizer, libero.optimizer)
                self.assertEqual(q5.keep_period, libero.keep_period)
                self.assertEqual(q5.save_interval, libero.save_interval)
                self.assertEqual(q5.model.max_token_len, libero.model.max_token_len)
                self.assertEqual(q5.model.action_dim, 38)
                self.assertTrue(q5.model.discrete_state_input)
                self.assertEqual(q5.model.action_horizon, 10)
        self.assertEqual(build_config(Paths.create(tempfile.gettempdir()), profile="finetune").num_train_steps, 1000)


class Q5WeightMigrationTests(unittest.TestCase):
    def setUp(self):
        self.source = {"trunk": {"kernel": np.full((4, 4), 7, dtype=np.float32)},
                       "action_in_proj": {"kernel": np.ones((32, 4), dtype=np.float32), "bias": np.ones(4, dtype=np.float32)},
                       "action_out_proj": {"kernel": np.ones((4, 32), dtype=np.float32), "bias": np.ones(32, dtype=np.float32)}}
        s = lambda shape: jax.ShapeDtypeStruct(shape, np.float32)
        self.target = {"trunk": {"kernel": s((4, 4)), "lora_a": s((4, 2))},
                       "action_in_proj": {"kernel": s((38, 4)), "bias": s((4,))},
                       "action_out_proj": {"kernel": s((4, 38)), "bias": s((38,))}}

    def test_preserves_trunk_and_defers_all_projections_and_lora_to_initializer(self):
        result = merge_trunk(self.source, self.target)
        np.testing.assert_array_equal(result["trunk"]["kernel"], self.source["trunk"]["kernel"])
        flat_result = flax.traverse_util.flatten_dict(result, sep="/")
        flat_target = flax.traverse_util.flatten_dict(self.target, sep="/")
        for key in PROJECTION_KEYS | {"trunk/lora_a"}:
            self.assertIs(flat_result[key], flat_target[key])
        self.assertEqual(self.source["action_out_proj"]["kernel"].shape, (4, 32))

    def test_loader_writes_explicit_migration_and_freeze_inventory(self):
        from pipeline.q5.config import build_config
        self.source["PaliGemma"] = {"llm": {"qkv_1": {"kernel": np.ones((2, 2), np.float32)}}}
        self.target["PaliGemma"] = {"llm": {"qkv_1": {"kernel": jax.ShapeDtypeStruct((2, 2), np.float32),
                                                     "lora_a": jax.ShapeDtypeStruct((2, 1), np.float32)}}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            config = build_config(Paths.create(directory))
            with mock.patch("openpi.models.model.restore_params", return_value=self.source):
                Q5WeightLoader("fake-local-checkpoint", str(output), config.trainable_filter).load(self.target)
            report = json.loads(output.read_text())
            rows = {row["path"]: row for row in report["parameters"]}
            self.assertEqual(set(report["reinitialized_projection_paths"]), PROJECTION_KEYS)
            self.assertEqual(rows["action_out_proj/kernel"]["initialization"], "model_rng_action_projection")
            self.assertEqual(rows["PaliGemma/llm/qkv_1/lora_a"]["initialization"], "model_rng_new_lora")
            self.assertEqual(rows["PaliGemma/llm/qkv_1/kernel"]["initialization"], "pretrained_checkpoint")
            self.assertFalse(rows["PaliGemma/llm/qkv_1/kernel"]["trainable"])

    def test_missing_trunk_is_not_silently_random_initialized(self):
        bad = copy.deepcopy(self.source)
        del bad["trunk"]
        with self.assertRaisesRegex(ValueError, "Missing pretrained trunk"):
            merge_trunk(bad, self.target)

    def test_wrong_trunk_shape_is_rejected(self):
        bad = copy.deepcopy(self.source)
        bad["trunk"]["kernel"] = np.zeros((5, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "trunk shape mismatch"):
            merge_trunk(bad, self.target)

    def test_native_38_checkpoint_is_not_mistaken_for_base_migration(self):
        bad = copy.deepcopy(self.source)
        bad["action_out_proj"]["kernel"] = np.ones((4, 38), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "migration shape"):
            merge_trunk(bad, self.target)

    def test_unknown_checkpoint_key_is_rejected(self):
        bad = copy.deepcopy(self.source)
        bad["unexpected"] = np.zeros(1, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Unexpected checkpoint keys"):
            merge_trunk(bad, self.target)


class Q5PopulationGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = Paths.create(self.directory.name)
        m = {"status": "PASS"}
        tests = set(choose_test_ids(42))
        validation = set(choose_validation_ids(tests))
        m["locked_test_source_episode_ids"] = sorted(tests)
        for role, ids, root in [("train", sorted(set(range(65)) - tests - validation), self.paths.train),
                                ("validation", sorted(validation), self.paths.validation),
                                ("test", sorted(tests), self.paths.test)]:
            (root / "meta").mkdir(parents=True)
            (root / "meta/info.json").write_text(json.dumps({"splits": {role: f"0:{len(ids)}"},
                    "total_episodes": len(ids), "total_frames": len(ids)}))
            records = []
            for new, old in enumerate(ids):
                relative = f"episode_{new:06d}.parquet"
                (root / relative).write_bytes(f"fixture-{old}".encode())
                records.append(dict(source_episode_index=old, output_episode_index=new, length=1,
                                    data_path=relative, sha256=sha256(root / relative)))
            m[role] = records
        self.paths.split_manifest.write_text(json.dumps(m))
        self.paths.stats.mkdir(parents=True)
        (self.paths.stats / "norm_stats.json").write_text("{}")
        self.report = dict(status="PASS", training_eligible=True, test_episodes_used=0, validation_episodes_used=0,
                           episodes_used=50, frames_used=50, selected_episode_ids=list(range(50)),
                           split="train", action_horizon=10, joint_dimension=38,
                           action_representation="absolute_joint_targets", delta_transform=False,
                           source_info_sha256=sha256(self.paths.train / "meta/info.json"),
                           split_manifest_sha256=sha256(self.paths.split_manifest),
                           sha256=sha256(self.paths.stats / "norm_stats.json"), joint_names=list(JOINT_NAMES),
                           source_parquet_fingerprints=[{"path": r["data_path"], "sha256": r["sha256"]} for r in m["train"]])
        self.save_report()

    def save_report(self):
        (self.paths.stats / "statistics_run.json").write_text(json.dumps(self.report))

    def test_training_only_record_is_accepted(self):
        self.assertEqual(require_stats(self.paths)["episodes_used"], 50)

    def test_old_55_statistics_are_rejected(self):
        self.report["episodes_used"] = 55
        self.save_report()
        with self.assertRaisesRegex(ValueError, "statistics population"):
            require_stats(self.paths)

    def test_validation_observations_in_statistics_are_rejected(self):
        self.report["validation_episodes_used"] = 1
        self.save_report()
        with self.assertRaisesRegex(ValueError, "Validation episodes"):
            require_stats(self.paths)

    def test_full_65_statistics_are_rejected(self):
        self.report["episodes_used"] = 65
        self.save_report()
        with self.assertRaisesRegex(ValueError, "statistics population"):
            require_stats(self.paths)

    def test_test_observations_in_statistics_are_rejected(self):
        self.report["test_episodes_used"] = 1
        self.save_report()
        with self.assertRaisesRegex(ValueError, "Test episodes"):
            require_stats(self.paths)

    def test_overlapping_split_is_rejected(self):
        m = json.loads(self.paths.split_manifest.read_text())
        m["test"][0]["source_episode_index"] = m["train"][0]["source_episode_index"]
        self.paths.split_manifest.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_split(self.paths)

    def test_changed_training_parquet_is_rejected(self):
        (self.paths.train / "episode_000000.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Changed train Parquet"):
            require_stats(self.paths)


class Q5ScheduleTests(unittest.TestCase):
    def test_warmup_only_finetune_requires_explicit_continuation(self):
        from pipeline.q5.schedule import validate_schedule
        with self.assertRaisesRegex(ValueError, "ends during warmup"):
            validate_schedule("finetune", 1000, 1000, 30000)
        plan = validate_schedule("finetune", 1000, 1000, 30000, True)
        self.assertTrue(plan["segment_entirely_in_warmup"])
        self.assertFalse(plan["schedule_endpoint_reached_by_segment"])

    def test_standalone_explicit_schedule_can_finish(self):
        from pipeline.q5.schedule import validate_schedule
        plan = validate_schedule("finetune", 1000, 100, 1000)
        self.assertFalse(plan["segment_entirely_in_warmup"])
        self.assertTrue(plan["schedule_endpoint_reached_by_segment"])

    def test_bad_schedule_is_rejected(self):
        from pipeline.q5.schedule import validate_schedule
        for segment, warmup, decay in [(0, 10, 100), (100, -1, 100), (100, 100, 100), (101, 10, 100)]:
            with self.subTest(segment=segment, warmup=warmup, decay=decay), self.assertRaises(ValueError):
                validate_schedule("smoke", segment, warmup, decay)


class Q5WholeParameterAuditTests(unittest.TestCase):
    def test_action_expert_non_lora_frozen_and_every_leaf_listed(self):
        from pipeline.q5.config import build_config
        from pipeline.q5.parameter_audit import audit_parameters
        s = lambda shape: jax.ShapeDtypeStruct(shape, np.float32)
        tree = {"PaliGemma": {"llm": {"qkv": {"kernel": s((2, 2)), "lora_a": s((2, 1))},
                                      "qkv_1": {"kernel": s((2, 2)), "lora_b": s((1, 2))}},
                             "img": {"kernel": s((2, 2))}}, "action_in_proj": {"bias": s((2,))}}
        report = audit_parameters(tree, build_config(Paths.create(tempfile.gettempdir())).trainable_filter)
        self.assertEqual(len(report["parameters"]), 6)
        self.assertEqual(report["action_expert_non_lora_leaves_checked"], 1)
        rows = {r["path"]: r for r in report["parameters"]}
        self.assertFalse(rows["PaliGemma/llm/qkv_1/kernel"]["trainable"])
        self.assertTrue(rows["PaliGemma/llm/qkv_1/lora_b"]["trainable"])
        self.assertTrue(rows["PaliGemma/img/kernel"]["trainable"])
        import flax.nnx as nnx
        with self.assertRaisesRegex(ValueError, "freeze state"):
            audit_parameters(tree, nnx.Param)


if __name__ == "__main__":
    unittest.main()
