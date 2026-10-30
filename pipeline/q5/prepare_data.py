"""Create 50/5/10 copies, preserve the original test identities, compute train-only stats."""
import argparse
import json
from pathlib import Path
import random

from pipeline.q5.runtime import Paths, SPLIT_METADATA_FILES, prepare_environment, require, sha256, validate_split
from pipeline.q5.schema import JOINT_NAMES


def choose_test_ids(seed=42):
    """One seeded sample from each of ten consecutive collection-order strata."""
    rng = random.Random(seed)
    return sorted(rng.choice(list(range(i * 65 // 10, (i + 1) * 65 // 10))) for i in range(10))


def choose_validation_ids(test_ids, seed=43):
    candidates = sorted(set(range(65)) - set(test_ids))
    require(len(candidates) == 55, "Expected 55 non-test candidates")
    rng = random.Random(seed)
    return sorted(rng.choice(candidates[i * 55 // 5:(i + 1) * 55 // 5]) for i in range(5))


def read_original_episode_ids(source, episode_count):
    """Optional cleaning history; never invent pre-cleaning episode identities."""
    path = source / "cleaning_manifest.json"
    if not path.is_file():
        return {eid: None for eid in range(episode_count)}
    rows = json.loads(path.read_text())["episode_mapping"]
    require(len(rows) == episode_count, "Cleaning history must cover every source episode")
    mapping = {}
    for row in rows:
        eid, original = row["new_episode_index"], row["original_episode_index"]
        require(type(eid) is int and 0 <= eid < episode_count,
                "Invalid source episode ID in cleaning history")
        require(eid not in mapping, "Duplicate source episode ID in cleaning history")
        require(type(original) is int and original >= 0, "Invalid original episode ID in cleaning history")
        mapping[eid] = original
    require(len(set(mapping.values())) == episode_count, "Duplicate original episode ID in cleaning history")
    return mapping


def prepare_split(paths, source, *, seed=42):
    from pipeline.data.exclude_episodes import run
    require(not any(p.exists() for p in (paths.train, paths.validation, paths.test, paths.split_manifest)), "Split output already exists; use --stats-only or inspect existing data")
    info = json.loads((source / "meta/info.json").read_text())
    require(info["total_episodes"] == 65, "Expected the cleaned 65-episode source")
    for key in ["action", "observation.state"]:
        require(info["features"][key]["names"] == list(JOINT_NAMES), f"Unexpected Q5 joint mapping: {key}")
    test_ids = set(choose_test_ids(seed))
    previous = paths.workspace / "datasets/q5/pick_orange_to_basket_v1_split.json"
    if previous.is_file():
        old_manifest = json.loads(previous.read_text())
        locked = {r["source_episode_index"] for r in old_manifest["test"]}
        require(len(locked) == 10, "Invalid previous test population")
        test_ids = locked
    validation_ids = set(choose_validation_ids(test_ids))
    train_ids = set(range(65)) - test_ids - validation_ids
    source_map = read_original_episode_ids(source, info["total_episodes"])
    cleaning_history = source / "cleaning_manifest.json"
    marker = paths.split_manifest.with_suffix(".INCOMPLETE")
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x") as f:
        f.write("Split preparation in progress; do not train.\n")
    manifest = dict(status="PASS", seed=seed, validation_seed=43, metadata_sha256={},
                    locked_test_source_episode_ids=sorted(test_ids),
                    previous_split_sha256=sha256(previous) if previous.is_file() else None,
                    selection="original ten test episodes locked; five validation samples stratified within the remaining 55; no outcome-based selection",
                    source=str(source), source_info_sha256=sha256(source / "meta/info.json"),
                    cleaning_history_available=cleaning_history.is_file(),
                    cleaning_manifest_sha256=sha256(cleaning_history) if cleaning_history.is_file() else None,
                    episode_identity="source_episode_index identifies the current source dataset; "
                                     "original_episode_index is null when pre-cleaning history is unavailable")
    for role, destination, included in [("train", paths.train, train_ids), ("validation", paths.validation, validation_ids), ("test", paths.test, test_ids)]:
        excluded = set(range(65)) - included
        copied = run(source, destination, excluded)
        dst_info_path = destination / "meta/info.json"
        dst_info = json.loads(dst_info_path.read_text())
        dst_info["splits"] = {role: f"0:{len(copied['episode_mapping'])}"}
        dst_info_path.write_text(json.dumps(dst_info, indent=2) + "\n")
        records = []
        for row in copied["episode_mapping"]:
            src_id, dst_id = row["original_episode_index"], row["new_episode_index"]
            data_path = dst_info["data_path"].format(episode_index=dst_id, episode_chunk=dst_id // dst_info["chunks_size"])
            timing = json.loads((destination / f"meta/timing/episode_{dst_id:06d}.json").read_text())
            assets = [{"path": f"meta/timing/episode_{dst_id:06d}.json",
                       "sha256": sha256(destination / f"meta/timing/episode_{dst_id:06d}.json")}]
            for key, feature in dst_info["features"].items():
                if feature["dtype"] == "video":
                    video = dst_info["video_path"].format(episode_index=dst_id, episode_chunk=dst_id // dst_info["chunks_size"], video_key=key)
                    assets.append({"path": video, "sha256": sha256(destination / video)})
            records.append(dict(source_episode_index=src_id, original_episode_index=source_map[src_id],
                                output_episode_index=dst_id, length=row["length"], data_path=data_path,
                                sha256=sha256(destination / data_path), source_bag=timing["source_bag"], assets=assets))
        manifest[role] = records
        manifest["metadata_sha256"][role] = {name: sha256(destination / name) for name in SPLIT_METADATA_FILES}
    paths.split_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    marker.unlink()
    try:
        return validate_split(paths)
    except Exception:
        marker.write_text("Split validation failed; do not train.\n")
        raise


def compute_stats(paths):
    validate_split(paths)
    from pipeline.data.compute_q5_norm_stats import main as compute
    compute(["--data-root", str(paths.train), "--openpi-root", str(paths.workspace / "openpi"),
             "--output-dir", str(paths.stats), "--action-horizon", "10", "--quantiles", "exact"])
    report_path = paths.stats / "statistics_run.json"
    report = json.loads(report_path.read_text())
    report.update(training_eligible=True, split_manifest_sha256=sha256(paths.split_manifest),
                  test_episodes_used=0, validation_episodes_used=0, state_input="38-dimensional normalized state tokens")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    from pipeline.q5.runtime import require_stats
    require_stats(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stats-only", action="store_true")
    mode.add_argument("--skip-stats", action="store_true")
    args = parser.parse_args()
    paths = Paths.create(args.workspace)
    prepare_environment(paths, cpu=True)
    if not args.stats_only:
        source = (args.source or paths.workspace / "datasets/q5/pick_orange_to_basket_no_ep0").resolve()
        prepare_split(paths, source, seed=args.seed)
    if not args.skip_stats:
        compute_stats(paths)
    manifest = validate_split(paths)
    print(json.dumps({"status": "PASS", "train_episodes": len(manifest["train"]), "validation_episodes": len(manifest["validation"]), "test_episodes": len(manifest["test"]),
                      "train": str(paths.train), "validation": str(paths.validation), "test": str(paths.test), "manifest": str(paths.split_manifest),
                      "stats": str(paths.stats) if not args.skip_stats else None}, indent=2))


if __name__ == "__main__":
    main()
