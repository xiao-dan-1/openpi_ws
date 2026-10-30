"""Audit Q5 data into a new external directory; exit 1 on errors, 0 on success.

Requires PyAV, NumPy, PyArrow and Pillow. Source data is never modified.
An INCOMPLETE marker remains if the audit cannot finish.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distribution(values):
    a = np.asarray(values)
    return dict(zip(["min", "p01", "median", "p99", "max"],
                    np.percentile(a, [0, 1, 50, 99, 100]).tolist())) if a.size else {}


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def source_qc(root):
    """Summarize converter provenance; these counters are not raw-bag revalidation."""
    rows = []
    topics = {}
    for path in sorted((root / "meta/timing").glob("episode_*.json")):
        d = json.loads(path.read_text())
        qc = d.get("qc", {})
        row = {"episode": int(path.stem.split("_")[-1]),
               "output_images": len(d.get("relative_image_stamps", [])),
               "source_images": qc.get("n_img", 0)}
        for key in ["no_state", "no_action", "stale_state", "stale_action", "disorder", "age_a_future"]:
            row[key] = qc.get(key, 0)
        for key in ["age_s_min", "age_s_max", "age_a_min", "age_a_max", "back_max"]:
            row[key] = qc.get(key, 0)
        rows.append(row)
        for topic, values in qc.get("gaps", {}).items():
            topics.setdefault(topic, []).append(values)
    return dict(provenance="Counters from converter timing sidecars; raw bags were not re-read.",
                totals={k: sum(r[k] for r in rows) for k in ["output_images", "source_images", "no_state", "no_action", "stale_state", "stale_action", "disorder", "age_a_future"]},
                maxima={k: max((r[k] for r in rows), default=None) for k in ["age_s_max", "age_a_max", "back_max"]},
                minima={k: min((r[k] for r in rows), default=None) for k in ["age_s_min", "age_a_min"]},
                topic_gap_raw_values=topics, episodes=rows)


def audit(root, out):
    """Read data, write review artifacts, and return a completed audit report."""
    info = json.loads((root / "meta/info.json").read_text())
    episodes = jsonl(root / "meta/episodes.jsonl")
    tasks = jsonl(root / "meta/tasks.jsonl")
    stats_rows = jsonl(root / "meta/episodes_stats.jsonl")
    stored_stats = {e["episode_index"]: e["stats"] for e in stats_rows}
    fps = info["fps"]
    errors, warnings, rows, joints, jumps = [], [], [], [], []
    arrays = {"action": [], "observation.state": []}
    differences = {k: [] for k in arrays}
    lags = {group: {lag: [] for lag in range(-15, 16)}
            for group in ["left_arm", "right_arm", "left_hand", "right_hand"]}
    groups = {"left_arm": slice(0, 7), "right_arm": slice(7, 14),
              "left_hand": slice(14, 26), "right_hand": slice(26, 38)}
    sample_thumbs, ends, expected_files, all_indices = [], [], set(), []
    stats_mismatches, schema_types, timing_keys = [], set(), set()
    video_key = "observation.images.image_head"
    def require(ok, message):
        if not ok:
            errors.append(message)
    require(len(episodes) == info["total_episodes"], "episode metadata count")
    require(len(tasks) == info["total_tasks"], "task metadata count")
    require(len(stats_rows) == len(episodes), "episode stats count")
    require([e["episode_index"] for e in episodes] == list(range(info["total_episodes"])),
            "episode indices missing, duplicated, or unordered")
    task_map = {t["task_index"]: t["task"] for t in tasks}
    for episode in episodes:
        eid, nmeta = episode["episode_index"], episode["length"]
        fmt = dict(episode_index=eid, episode_chunk=eid // info["chunks_size"], video_key=video_key)
        data_file, video_file = root / info["data_path"].format(**fmt), root / info["video_path"].format(**fmt)
        expected_files.update([data_file, video_file])
        if not data_file.is_file() or not video_file.is_file():
            errors.append(f"episode {eid}: missing parquet/video")
            continue
        table = pq.read_table(data_file)
        n = len(table)
        schema_types.add(str(table.schema))
        require(n == nmeta and n > 1, f"episode {eid}: row count {n} vs {nmeta}")
        require(table.null_count == 0 if hasattr(table, "null_count") else
                all(c.null_count == 0 for c in table.columns), f"episode {eid}: null cells")
        require(np.array_equal(table["frame_index"].to_numpy(), np.arange(n)), f"episode {eid}: frame_index")
        require(np.all(table["episode_index"].to_numpy() == eid), f"episode {eid}: episode_index")
        tids = set(table["task_index"].to_pylist())
        require(tids <= set(task_map), f"episode {eid}: unknown task")
        require({task_map[t] for t in tids if t in task_map} == set(episode["tasks"]), f"episode {eid}: task mismatch")
        all_indices.extend(table["index"].to_pylist())
        ts = table["timestamp"].to_numpy().astype(float)
        require(np.all(np.diff(ts) > 0), f"episode {eid}: nonmonotonic timestamps")
        timestamp_error = float(np.max(np.abs(ts - np.arange(n) / fps)))
        require(timestamp_error < 1e-4, f"episode {eid}: timestamps vs nominal FPS")
        row = dict(episode=eid, frames=n, duration_s=n / fps, timestamp_max_error_s=timestamp_error)
        vals = {}
        for key in arrays:
            a = np.asarray(table[key].to_pylist(), dtype=float)
            require(a.shape == (n, 38), f"episode {eid}: {key} shape {a.shape}")
            require(np.isfinite(a).all(), f"episode {eid}: {key} NaN/Inf")
            require(str(table.schema.field(key).type) == "fixed_size_list<element: float>[38]",
                    f"episode {eid}: {key} dtype")
            arrays[key].append(a)
            vals[key] = a
            delta = np.diff(a, axis=0)
            differences[key].append(delta)
            short = "action" if key == "action" else "state"
            row[f"{short}_exact_repeat_fraction"] = float(np.mean(np.all(delta == 0, axis=1)))
            row[f"{short}_max_abs_step"] = float(np.max(np.abs(delta)))
            for flat in np.argsort(np.abs(delta).ravel())[-3:][::-1]:
                frame, dim = np.unravel_index(flat, delta.shape)
                jumps.append(dict(episode=eid, feature=key, frame=int(frame + 1),
                                  joint=info["features"][key]["names"][dim],
                                  before=float(a[frame, dim]), after=float(a[frame + 1, dim]),
                                  abs_step=float(abs(delta[frame, dim]))))
            stored = stored_stats.get(eid, {}).get(key, {})
            for stat, computed in [("min", a.min(0)), ("max", a.max(0)),
                                   ("mean", a.mean(0)), ("std", a.std(0))]:
                if stat in stored and not np.allclose(stored[stat], computed, atol=2e-5, rtol=2e-4):
                    stats_mismatches.append(dict(episode=eid, feature=key, stat=stat,
                                                max_abs_error=float(np.max(np.abs(np.asarray(stored[stat]) - computed)))))
                    errors.append(f"episode {eid}: stored {key}.{stat} differs from data")
            if "count" in stored:
                require(np.all(np.asarray(stored["count"]) == n), f"episode {eid}: stats count {key}")
        for name, sl in groups.items():
            state, action = vals["observation.state"][:, sl], vals["action"][:, sl]
            row[f"{name}_state_range_max"] = float(np.ptp(state, axis=0).max())
            row[f"{name}_action_state_mae"] = float(np.abs(state - action).mean())
            for lag in lags[name]:
                # lag > 0: present command compared with future measured state.
                x, y = (action[:-lag], state[lag:]) if lag > 0 else ((action[-lag:], state[:lag]) if lag < 0 else (action, state))
                if len(x):
                    lags[name][lag].append(float(np.mean(np.abs(x - y))))
        timing_file = root / "meta/timing" / f"episode_{eid:06d}.json"
        require(timing_file.exists(), f"episode {eid}: missing source timing")
        if timing_file.exists():
            timing = json.loads(timing_file.read_text())
            timing_keys.update(timing.keys())
            stamps = np.asarray(timing.get("relative_image_stamps", []), dtype=float)
            require(len(stamps) == n, f"episode {eid}: source image timestamp count")
            if len(stamps) == n:
                dt = np.diff(stamps)
                require(np.isfinite(stamps).all() and np.all(dt > 0), f"episode {eid}: source time invalid")
                row.update(source_dt_min_s=float(dt.min()), source_dt_max_s=float(dt.max()),
                           source_vs_video_max_drift_s=float(np.max(np.abs(stamps - np.arange(n) / fps))))
        wanted = sorted(set(np.rint(np.linspace(0, n - 1, 6)).astype(int)))
        thumbs, brightness, blur, pts, video_deltas = {}, [], [], [], []
        seen_hashes, previous_gray, longest_repeat, repeat_run = set(), None, 0, 0
        exact_repeats = black = white = decoded = 0
        try:
            with av.open(str(video_file)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                row.update(video_header_frames=stream.frames, video_fps=float(stream.average_rate),
                           video_width=stream.width, video_height=stream.height)
                require((stream.height, stream.width, 3) == tuple(info["features"][video_key]["shape"]), f"episode {eid}: video shape")
                require(abs(float(stream.average_rate) - fps) < 1e-6, f"episode {eid}: video FPS")
                for frame_i, frame in enumerate(container.decode(stream)):
                    decoded += 1
                    pts.append(float(frame.pts * frame.time_base) if frame.pts is not None else None)
                    gray = frame.reformat(width=160, height=120, format="gray").to_ndarray()
                    brightness.append(float(gray.mean()))
                    # Simple gradient energy is a relative sharpness indicator, not a rejection rule.
                    blur.append(float(np.abs(np.diff(gray.astype(float), axis=0)).mean() +
                                      np.abs(np.diff(gray.astype(float), axis=1)).mean()))
                    black += int(np.mean(gray < 8) > 0.99)
                    white += int(np.mean(gray > 247) > 0.99)
                    h = hashlib.sha256(gray.tobytes()).digest()
                    seen_hashes.add(h)
                    if previous_gray is not None:
                        is_repeat = np.array_equal(gray, previous_gray)
                        exact_repeats += int(is_repeat)
                        repeat_run = repeat_run + 1 if is_repeat else 0
                        longest_repeat = max(longest_repeat, repeat_run)
                        video_deltas.append(float(np.abs(gray.astype(float) - previous_gray).mean()))
                    previous_gray = gray
                    if frame_i in wanted:
                        thumbs[frame_i] = frame.to_image().resize((320, 240))
        except Exception as exc:
            errors.append(f"episode {eid}: video decode {exc}")
        require(decoded == n, f"episode {eid}: decoded video {decoded} vs data {n}")
        valid_pts = np.array([p for p in pts if p is not None])
        require(len(valid_pts) == n, f"episode {eid}: video missing PTS")
        if len(valid_pts) == n:
            require(np.all(np.diff(valid_pts) > 0), f"episode {eid}: nonmonotonic video PTS")
            row["video_pts_max_error_s"] = float(np.max(np.abs(valid_pts - ts)))
            require(row["video_pts_max_error_s"] < 1 / fps / 2, f"episode {eid}: video PTS/data misalignment")
        row.update(decoded_video_frames=decoded, black_frames=black, white_frames=white,
                   repeated_video_frames=exact_repeats, longest_identical_video_run_frames=longest_repeat + 1,
                   unique_video_thumbnails=len(seen_hashes), brightness_mean=float(np.mean(brightness)) if brightness else None,
                   brightness_min=float(np.min(brightness)) if brightness else None,
                   sharpness_median=float(np.median(blur)) if blur else None,
                   video_change_median=float(np.median(video_deltas)) if video_deltas else None)
        if black or white:
            warnings.append(f"episode {eid}: {black} black, {white} white frames")
        if longest_repeat >= fps:
            warnings.append(f"episode {eid}: >=1s identical decoded thumbnails")
        sample_thumbs.append((eid, [(i, thumbs[i]) for i in wanted if i in thumbs]))
        if thumbs:
            ends.append((eid, thumbs[max(thumbs)]))
        rows.append(row)
        print(f"episode {eid:02d}: {n} rows, {decoded} decoded frames, errors={len(errors)}", flush=True)
    require(sum(r["frames"] for r in rows) == info["total_frames"], "total_frames")
    require(np.array_equal(all_indices, np.arange(info["total_frames"])), "global index not consecutive/unique")
    actual_data = set(root.glob("data/**/*.parquet"))
    actual_video = set(root.glob("videos/**/*.mp4"))
    require(len(actual_data) == len(episodes), "unexpected parquet count")
    require(len(actual_video) == info["total_videos"], "unexpected video count")
    unreferenced = sorted(str(p.relative_to(root)) for p in (actual_data | actual_video) - expected_files)
    require(not unreferenced, "unreferenced data/video files")
    for key, chunks in arrays.items():
        if not chunks:
            continue
        a, delta = np.concatenate(chunks), np.concatenate(differences[key])
        for dim, name in enumerate(info["features"][key]["names"]):
            joints.append(dict(feature=key, joint=name, minimum=float(a[:, dim].min()),
                               maximum=float(a[:, dim].max()), std=float(a[:, dim].std()),
                               p01=float(np.percentile(a[:, dim], 1)), p99=float(np.percentile(a[:, dim], 99)),
                               unique_values=len(np.unique(a[:, dim])),
                               step_p99=float(np.percentile(np.abs(delta[:, dim]), 99)),
                               step_max=float(np.abs(delta[:, dim]).max()),
                               exact_repeat_fraction=float(np.mean(delta[:, dim] == 0))))
    for start in range(0, len(sample_thumbs), 11):
        subset = sample_thumbs[start:start + 11]
        sheet = Image.new("RGB", (6 * 320, len(subset) * 264), "#17202b")
        draw = ImageDraw.Draw(sheet)
        for r, (eid, thumbs) in enumerate(subset):
            for c, (frame_i, thumb) in enumerate(thumbs):
                sheet.paste(thumb, (c * 320, r * 264 + 24))
                draw.text((c * 320 + 7, r * 264 + 5), f"ep {eid:02d} | frame {frame_i} | {frame_i / fps:.2f}s", fill="white")
        sheet.save(out / f"episodes_{start:02d}_{start + len(subset) - 1:02d}.jpg", quality=90)
    if ends:
        sheet = Image.new("RGB", (6 * 240, ((len(ends) + 5) // 6) * 200), "#17202b")
        draw = ImageDraw.Draw(sheet)
        for i, (eid, thumb) in enumerate(ends):
            x, y = i % 6 * 240, i // 6 * 200
            sheet.paste(thumb.resize((240, 180)), (x, y + 20))
            draw.text((x + 5, y + 3), f"episode {eid:02d} - final frame", fill="white")
        sheet.save(out / "all_final_frames.jpg", quality=92)
    report = dict(status="FAIL" if errors else "PASS", root=str(root), metadata=info, errors=errors, warnings=warnings,
                  source_qc=source_qc(root),
                  episodes_checked=len(rows), decoded_video_frames=sum(r["decoded_video_frames"] for r in rows),
                  total_duration_s=info["total_frames"] / fps, duration_s=distribution([r["duration_s"] for r in rows]),
                  schema_variants=len(schema_types), schemas=sorted(schema_types),
                  timing_keys=sorted(timing_keys), metadata_stat_mismatches=stats_mismatches,
                  unreferenced_files=unreferenced,
                  constant_features=[j for j in joints if j["unique_values"] == 1],
                  largest_steps=sorted(jumps, key=lambda j: j["abs_step"], reverse=True)[:40],
                  lag_mae_equal_episode_weight={g: {str(lag): float(np.mean(v)) if v else None for lag, v in data.items()}
                                                for g, data in lags.items()},
                  source_dt_s=distribution([r["source_dt_max_s"] for r in rows if "source_dt_max_s" in r]),
                  source_max_drift_s=max((r["source_vs_video_max_drift_s"] for r in rows
                                          if "source_vs_video_max_drift_s" in r), default=None),
                  video_totals={k: sum(r[k] for r in rows) for k in ["black_frames", "white_frames", "repeated_video_frames"]},
                  limitations=["No robot joint limits or action-unit contract supplied; range and step sizes are descriptive.",
                               "Video PTS and source image timestamps checked; raw joint/command timestamps unavailable unless recorded in timing sidecars.",
                               "Visual task success requires human/video review; automated metrics do not certify success.",
                               "Exact repeats and brightness use 160x120 grayscale decoded thumbnails."])
    (out / "quality_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out / "episodes.csv", rows)
    write_csv(out / "joint_statistics.csv", joints)
    write_csv(out / "largest_steps.csv", sorted(jumps, key=lambda j: j["abs_step"], reverse=True))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path, help="New report directory outside the dataset; never overwritten")
    args = parser.parse_args(argv)
    root, out = args.root.resolve(), args.output.resolve()
    if out == root or root in out.parents:
        parser.error("Output must be outside the source dataset")
    if not (root / "meta/info.json").is_file():
        parser.error(f"Missing dataset metadata: {root / 'meta/info.json'}")
    try:
        out.mkdir(parents=True)
    except FileExistsError:
        parser.error(f"Output already exists; choose a new directory: {out}")
    marker = out / "INCOMPLETE"
    marker.write_text("Audit did not finish; do not use partial reports.\n", encoding="utf-8")
    try:
        report = audit(root, out)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}\nIncomplete audit: {out}", file=sys.stderr)
        return 1
    marker.unlink()
    print(json.dumps({k: report[k] for k in ["status", "episodes_checked", "decoded_video_frames", "errors",
                     "warnings", "metadata_stat_mismatches", "constant_features", "video_totals",
                     "source_max_drift_s"]}, indent=2), flush=True)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    sys.exit(main())
