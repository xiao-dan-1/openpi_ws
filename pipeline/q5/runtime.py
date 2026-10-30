"""Workspace paths, provenance checks and offline environment; no model imports."""
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from pipeline.common.config import UPSTREAM_COMMIT, SUPPORTED_LOCAL_COMMITS
from pipeline.q5.memory import DEFAULT_MEMORY_FRACTION, configure_memory_fraction

NAME = "pi05_q5"
ASSET_ID = "q5_pick_orange_to_basket_v2_train"
TRAIN_ID = "q5/pick_orange_to_basket_v2_train"
VALIDATION_ID = "q5/pick_orange_to_basket_v2_validation"
TEST_ID = "q5/pick_orange_to_basket_v2_test"
SPLIT_COUNTS = {"train": 50, "validation": 5, "test": 10}
SPLIT_METADATA_FILES = ("meta/tasks.jsonl", "meta/episodes.jsonl")
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    require(revision in {UPSTREAM_COMMIT, *SUPPORTED_LOCAL_COMMITS}, f"Unreviewed OpenPI revision: {revision}")
    dirty = subprocess.check_output([
        "git", "-C", str(repo), "diff", "HEAD", "--name-only", "--", "src/openpi",
        "scripts/train.py", "scripts/compute_norm_stats.py"], text=True).strip()
    require(not dirty, f"Unreviewed OpenPI runtime edits: {dirty}")
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      TOKENIZERS_PARALLELISM="false", HF_LEROBOT_HOME=str(paths.workspace / "datasets"),
                      HF_DATASETS_CACHE=str(paths.workspace / ".cache/huggingface/q5"),
                      OPENPI_DATA_HOME=str(paths.workspace / "openpi_cache"),
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    os.environ.pop("LEROBOT_HOME", None)
    sys.dont_write_bytecode = True
    if cpu:
        os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    else:
        if allow_multiple_gpus:
            selectors = gpu.split(",") if gpu else []
            require(selectors and len(selectors) == len(set(selectors)) and
                    all(s.isdigit() or s.startswith("GPU-") for s in selectors), "Specify distinct server GPU selectors")
        else:
            require(gpu and (gpu.isdigit() or (gpu.startswith("GPU-") and "," not in gpu)), "Specify one server GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ.pop("JAX_PLATFORMS", None)
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
        metadata = m.get("metadata_sha256")
        metadata = metadata.get(role) if isinstance(metadata, dict) else None
        require(isinstance(metadata, dict) and
                all(isinstance(metadata.get(name), str) and len(metadata[name]) == 64
                    for name in SPLIT_METADATA_FILES),
                f"Missing {role} metadata fingerprints; preserve existing experiments and rebuild "
                "the split and statistics from a trusted source in a separate workspace; do not backfill hashes")
        require(not (root / "INCOMPLETE").exists(), f"Incomplete {role} dataset")
        info = json.loads((root / "meta/info.json").read_text())
        require(info["splits"] == {role: f"0:{len(records)}"}, f"Incorrect {role} split metadata")
        require(info["total_episodes"] == len(records) and info["total_frames"] == sum(r["length"] for r in records), f"Incorrect {role} totals")
        require([r["output_episode_index"] for r in records] == list(range(len(records))), f"Nonconsecutive {role} episodes")
        episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
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
            for name in SPLIT_METADATA_FILES:
                require(sha256(root / name) == metadata[name], f"Changed {role} metadata: {name}")
            for row in records:
                require(sha256(root / row["data_path"]) == row["sha256"], f"Changed {role} Parquet: {row['data_path']}")
                for asset in row.get("assets", []):
                    require(sha256(root / asset["path"]) == asset["sha256"], f"Changed {role} asset: {asset['path']}")
    return m


def require_stats(paths):
    validate_split(paths)
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
    manifest = json.loads(paths.split_manifest.read_text())
    require({r["path"] for r in report["source_parquet_fingerprints"]} == {r["data_path"] for r in manifest["train"]}, "Statistics file population differs from training split")
    for row in report["source_parquet_fingerprints"]:
        require(sha256(paths.train / row["path"]) == row["sha256"], "Statistics training data changed")
    return report


def official_train_module(paths):
    sys.path.insert(0, str(paths.workspace / "openpi"))
    module = importlib.import_module("scripts.train")
    require(Path(module.__file__).resolve() == paths.workspace / "openpi/scripts/train.py", "Unexpected official trainer")
    return module
