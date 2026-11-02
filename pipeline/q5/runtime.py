"""Workspace paths, provenance checks and offline environment; no model imports."""
import dataclasses
import json
from pathlib import Path
import sys

from pipeline.common import runtime as common_runtime
from pipeline.common.runtime import UPSTREAM_COMMIT, SUPPORTED_LOCAL_COMMITS, sha256
from pipeline.common.validation import require, gpu_selectors
from pipeline.q5.memory import DEFAULT_MEMORY_FRACTION, configure_memory_fraction

NAME = "pi05_q5"
ASSET_ID = "q5_pick_orange_to_basket_v2_train"
TRAIN_ID = "q5/pick_orange_to_basket_v2_train"
VALIDATION_ID = "q5/pick_orange_to_basket_v2_validation"
TEST_ID = "q5/pick_orange_to_basket_v2_test"
SPLIT_COUNTS = {"train": 50, "validation": 5, "test": 10}
FREEZE_POLICY = "model.get_freeze_filter() (same as LIBERO LoRA baseline)"
VIDEO_BACKEND = "pyav"


def _q5_default_video_backend():
    return VIDEO_BACKEND


def configure_video_backend():
    """Pin LeRobot's defaults in this Q5 process, including official loader calls.

    The pinned OpenPI DataConfig does not expose video_backend. LeRobot selects
    TorchCodec merely if its package exists, even when its native libraries fail
    to load. Patch both imported default selectors before constructing datasets;
    each resulting dataset carries 'pyav' when passed to DataLoader workers.
    No installed files or other Python processes are changed.
    """
    import av
    from lerobot.common.datasets import lerobot_dataset, video_utils
    video_utils.get_safe_default_codec = _q5_default_video_backend
    lerobot_dataset.get_safe_default_codec = _q5_default_video_backend
    return av.__version__


@dataclasses.dataclass(frozen=True)
class Paths:
    workspace: Path

    @classmethod
    def create(cls, workspace=None):
        return cls(Path(workspace).resolve() if workspace else Path(__file__).resolve().parents[2])

    @property
    def train(self):
        return self.workspace / "datasets" / TRAIN_ID

    @property
    def test(self):
        return self.workspace / "datasets" / TEST_ID

    @property
    def validation(self):
        return self.workspace / "datasets" / VALIDATION_ID

    @property
    def split_manifest(self):
        return self.workspace / "datasets/q5/pick_orange_to_basket_v2_split.json"

    @property
    def stats(self):
        return self.workspace / "assets" / NAME / ASSET_ID

    @property
    def base_params(self):
        return self.workspace / "openpi_cache/openpi-assets/checkpoints/pi05_base/params"


def prepare_environment(paths, *, cpu=True, gpu=None, allow_multiple_gpus=False,
                        memory_fraction=DEFAULT_MEMORY_FRACTION):
    repo = paths.workspace / "openpi"
    # Q5's exact reviewed SHA whitelist also supports shallow local checkouts.
    revision = common_runtime.verify_upstream_revision(repo, verify_ancestry=False)
    if not cpu:
        gpu = gpu_selectors(gpu or "")
        require(allow_multiple_gpus or "," not in gpu, "Specify one server GPU")
    common_runtime.prepare_environment(paths.workspace, cpu=cpu, gpu=gpu,
                                       cache_dir=paths.workspace / ".cache/huggingface/q5")
    if not cpu:
        configure_memory_fraction(memory_fraction)
    sys.path.insert(0, str(repo / "src"))
    configure_video_backend()
    return revision


def validate_split(paths, *, check_files=True):
    """Check population and episode boundaries; optionally verify file digests."""
    require(not paths.split_manifest.with_suffix(".INCOMPLETE").exists(), "Split preparation is incomplete")
    m = json.loads(paths.split_manifest.read_text())
    require(m["status"] == "PASS", "Split has not passed validation")
    populations = {}
    bags = set()
    for role, count in SPLIT_COUNTS.items():
        require(len(m[role]) == count, f"Expected exactly {count} {role} records")
        populations[role] = {r["source_episode_index"] for r in m[role]}
        require(len(populations[role]) == count, f"Duplicate {role} episode")
        for row in m[role]:
            if "source_bag" in row:
                require(row["source_bag"] not in bags, "Source bag appears in multiple records")
                bags.add(row["source_bag"])
    require(all(populations[a].isdisjoint(populations[b]) for a, b in
                [("train", "validation"), ("train", "test"), ("validation", "test")]), "Expected disjoint 50/5/10 split")
    require(set.union(*populations.values()) == set(range(65)), "Split does not cover all 65 cleaned episodes")
    require(populations["test"] == set(m["locked_test_source_episode_ids"]), "Locked test population changed")
    for role, root in [("train", paths.train), ("validation", paths.validation), ("test", paths.test)]:
        records = m[role]
        require(len({row["data_path"] for row in records}) == len(records), f"Duplicate {role} Parquet paths")
        require(not (root / "INCOMPLETE").exists(), f"Incomplete {role} dataset")
        info = json.loads((root / "meta/info.json").read_text())
        require(info["splits"] == {role: f"0:{len(records)}"}, f"Incorrect {role} split metadata")
        require(info["total_episodes"] == len(records) and info["total_frames"] == sum(r["length"] for r in records), f"Incorrect {role} totals")
        require([r["output_episode_index"] for r in records] == list(range(len(records))), f"Nonconsecutive {role} episodes")
        episode_path = root / "meta/episodes.jsonl"
        require(episode_path.is_file(), f"Missing {role} episode metadata: meta/episodes.jsonl")
        try:
            episodes = [json.loads(line) for line in episode_path.read_text().splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {role} episode metadata: meta/episodes.jsonl") from exc
        require(len(episodes) == len(records) and
                all(isinstance(row, dict) and type(row.get("episode_index")) is int and
                    type(row.get("length")) is int for row in episodes) and
                {row["episode_index"]: row["length"] for row in episodes} ==
                {row["output_episode_index"]: row["length"] for row in records},
                f"Incorrect {role} episode IDs/lengths in meta/episodes.jsonl")
        from pipeline.q5.schema import JOINT_NAMES
        for key in ["action", "observation.state"]:
            if "features" in info:
                require(info["features"][key]["names"] == list(JOINT_NAMES), f"Changed {role} joint order")
        if check_files:
            for row in records:
                require(sha256(root / row["data_path"]) == row["sha256"], f"Changed {role} Parquet: {row['data_path']}")
                for asset in row.get("assets", []):
                    require(sha256(root / asset["path"]) == asset["sha256"], f"Changed {role} asset: {asset['path']}")
    return m


def require_stats(paths, *, with_manifest=False):
    manifest = validate_split(paths)
    info_path = paths.train / "meta/info.json"
    info = json.loads(info_path.read_text())
    report = json.loads((paths.stats / "statistics_run.json").read_text())
    require(not (paths.stats / "INCOMPLETE").exists(), "Statistics incomplete")
    require(report["status"] == "PASS" and report.get("training_eligible") is True, "Training-only statistics required")
    require(report.get("test_episodes_used") == 0, "Test episodes must not be used in statistics")
    require(report.get("validation_episodes_used") == 0, "Validation episodes must not be used in statistics")
    require(report["episodes_used"] == SPLIT_COUNTS["train"] and report["frames_used"] == info["total_frames"], "Wrong statistics population")
    require(report["selected_episode_ids"] == list(range(SPLIT_COUNTS["train"])), "Statistics must cover exactly the training-copy episodes")
    require(report["split"] == "train" and report["action_horizon"] == 10 and report["joint_dimension"] == 38, "Statistics shape/split mismatch")
    require(report["action_representation"] == "absolute_joint_targets" and report["delta_transform"] is False, "Statistics action representation mismatch")
    require(report["source_info_sha256"] == sha256(info_path), "Training metadata changed")
    require(report["split_manifest_sha256"] == sha256(paths.split_manifest), "Split manifest changed")
    require(report["sha256"] == sha256(paths.stats / "norm_stats.json"), "Statistics file changed")
    from pipeline.q5.schema import JOINT_NAMES
    require(report["joint_names"] == list(JOINT_NAMES), "Statistics joint order mismatch")
    require({r["path"] for r in report["source_parquet_fingerprints"]} == {r["data_path"] for r in manifest["train"]}, "Statistics file population differs from training split")
    fingerprints = report["source_parquet_fingerprints"]
    require(len({row["path"] for row in fingerprints}) == len(fingerprints), "Duplicate statistics file fingerprints")
    verified = {row["data_path"]: row["sha256"] for row in manifest["train"]}
    for row in fingerprints:
        require(verified[row["path"]] == row["sha256"], "Statistics training data changed")
    return (report, manifest) if with_manifest else report


def official_train_module(paths):
    return common_runtime.load_official_script(paths.workspace, "train")
