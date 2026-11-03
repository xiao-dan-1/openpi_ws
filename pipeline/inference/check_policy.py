"""Send real Q5 or LIBERO dataset observations to the running checkpoint policy."""
import argparse
import io
import json
import os
import time
from pathlib import Path

from pipeline.inference.policy_rpc import prepare_client, read_server_metadata, verify_server_metadata
from pipeline.common.validation import experiment_name, require
from pipeline.common.reporting import emit_report

WORKSPACE = Path(__file__).resolve().parents[2]


def sample_requests(episodes):
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    from openpi_client import image_tools

    root = WORKSPACE / "datasets/libero"
    info = json.loads((root / "meta/info.json").read_text())
    tasks = {item["task_index"]: item["task"] for item in
             (json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines())}
    for episode in episodes:
        if not 0 <= episode < info["total_episodes"]:
            raise ValueError(f"Invalid episode: {episode}")
        relative = info["data_path"].format(episode_chunk=episode // info["chunks_size"],
                                             episode_index=episode)
        table = pq.read_table(root / relative, columns=["image", "wrist_image", "state", "task_index"])
        for frame in sorted({0, len(table) // 2, len(table) - 1}):
            row = table.slice(frame, 1).to_pylist()[0]
            request = {"observation/state": np.asarray(row["state"], dtype=np.float32),
                       "prompt": tasks[row["task_index"]]}
            for key in ("image", "wrist_image"):
                image = row[key]
                source = io.BytesIO(image["bytes"]) if image.get("bytes") is not None else root / image["path"]
                with Image.open(source) as decoded:
                    pixels = np.asarray(decoded.convert("RGB"))
                # Dataset images already have the training orientation. The 180-degree
                # flip in the simulator runner applies to simulator frames only.
                request[f"observation/{key}"] = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(pixels, 224, 224))
            yield episode, frame, request


def q5_sample_requests(episodes):
    import av
    import numpy as np
    import pyarrow.parquet as pq
    from pipeline.q5.schema import JOINT_NAMES

    root = WORKSPACE / "datasets/q5/pick_orange_to_basket_v2_validation"
    info = json.loads((root / "meta/info.json").read_text())
    require(info["features"]["observation.state"]["names"] == list(JOINT_NAMES), "Q5 dataset joint order differs")
    tasks = {row["task_index"]: row["task"] for row in
             map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
    for episode in episodes:
        require(0 <= episode < info["total_episodes"], f"Invalid Q5 validation episode: {episode}")
        indices = dict(episode_index=episode, episode_chunk=episode // info["chunks_size"])
        table = pq.read_table(root / info["data_path"].format(**indices),
                              columns=["observation.state", "task_index", "timestamp"])
        require(len(table) > 0, f"Empty Q5 validation episode: {episode}")
        selected = {0, len(table) // 2, len(table) - 1}
        video = root / info["video_path"].format(**indices, video_key="observation.images.image_head")
        decoded_count = 0
        with av.open(str(video)) as container:
            for frame, decoded in enumerate(container.decode(video=0)):
                decoded_count += 1
                if frame not in selected:
                    continue
                row = table.slice(frame, 1).to_pylist()[0]
                require(decoded.time is not None and abs(decoded.time - row["timestamp"]) < 0.5 / info["fps"],
                        f"Q5 image timestamp differs: episode={episode}, frame={frame}")
                state = np.asarray(row["observation.state"], dtype=np.float32)
                require(state.shape == (38,) and np.isfinite(state).all(), "Invalid Q5 observation state")
                yield episode, frame, {"state": state, "images": {"image_head": decoded.to_ndarray(format="rgb24")},
                                       "prompt": tasks[row["task_index"]]}
        require(decoded_count == len(table), f"Q5 video/frame count differs: episode={episode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["libero", "q5"], default="libero")
    parser.add_argument("--exp-name", required=True, type=experiment_name)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, nargs="+",
                        help="Q5 validation defaults to 0 2 4; LIBERO defaults to 0 846 1692")
    parser.add_argument("--output", type=Path, help="Optionally save the inference report; otherwise print results only")
    args = parser.parse_args()
    if args.step < 0 or not 1 <= args.port <= 65535 or (args.episodes and min(args.episodes) < 0):
        parser.error("Step/episodes must be nonnegative; port must be 1..65535")
    episodes = args.episodes if args.episodes is not None else ([0, 2, 4] if args.robot == "q5" else [0, 846, 1692])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    prepare_client(args.host)
    import numpy as np
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    metadata = read_server_metadata(args.host, args.port)
    identity = verify_server_metadata(metadata, args.exp_name, args.step, workspace=WORKSPACE, robot=args.robot)
    client = WebsocketClientPolicy(args.host, args.port)
    verify_server_metadata(client.get_server_metadata(), args.exp_name, args.step, workspace=WORKSPACE, robot=args.robot)
    samples = []
    requests = q5_sample_requests(episodes) if args.robot == "q5" else sample_requests(episodes)
    expected_shape = (10, 38) if args.robot == "q5" else (10, 7)
    for episode, frame, request in requests:
        print(f"Inferring episode={episode}, frame={frame} (first request may compile)...", flush=True)
        started = time.monotonic()
        response = client.infer(request)
        elapsed = time.monotonic() - started
        actions = np.asarray(response["actions"])
        if actions.shape != expected_shape or not np.isfinite(actions).all():
            raise ValueError(f"Invalid policy actions: {actions.shape}")
        samples.append({"episode": episode, "frame": frame, "actions_shape": list(actions.shape),
                        "elapsed_seconds": round(elapsed, 3),
                        "actions_min": float(actions.min()), "actions_max": float(actions.max())})
    report = {"status": "PASS", "robot": args.robot, "policy": identity, "samples_checked": len(samples), "samples": samples,
              "scope": "Real checkpoint WebSocket inference on dataset observations; not simulation or task success"}
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
