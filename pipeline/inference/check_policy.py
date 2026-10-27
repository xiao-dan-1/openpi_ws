"""Send real LIBERO dataset observations to the running checkpoint policy."""
import argparse
import io
import json
import os
import time

from pipeline.common.config import WORKSPACE
from pipeline.inference.policy_rpc import prepare_client, read_server_metadata, verify_server_metadata
from pipeline.common.validation import experiment_name


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True, type=experiment_name)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 846, 1692])
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    prepare_client(args.host)
    import numpy as np
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    metadata = read_server_metadata(args.host, args.port)
    identity = verify_server_metadata(metadata, args.exp_name, args.step)
    client = WebsocketClientPolicy(args.host, args.port)
    verify_server_metadata(client.get_server_metadata(), args.exp_name, args.step)
    samples = []
    for episode, frame, request in sample_requests(args.episodes):
        print(f"Inferring episode={episode}, frame={frame} (first request may compile)...", flush=True)
        started = time.monotonic()
        response = client.infer(request)
        elapsed = time.monotonic() - started
        actions = np.asarray(response["actions"])
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid policy actions: {actions.shape}")
        samples.append({"episode": episode, "frame": frame, "actions_shape": list(actions.shape),
                        "elapsed_seconds": round(elapsed, 3),
                        "actions_min": float(actions.min()), "actions_max": float(actions.max())})
    report = {"status": "PASS", "policy": identity, "samples_checked": len(samples), "samples": samples,
              "scope": "Real checkpoint WebSocket inference on dataset observations; not simulation or task success"}
    output = WORKSPACE / "runs/pi05_libero" / args.exp_name / f"inference_probe_{args.step}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
