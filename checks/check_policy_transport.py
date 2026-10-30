"""Real images over official WebSocket transport with a MOCK policy, no model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from pipeline.common.reporting import artifact_directory, emit_report


def observation_fingerprints(observation):
    import numpy as np

    fingerprints = {}
    for key, shape in (("observation/image", (224, 224, 3)),
                       ("observation/wrist_image", (224, 224, 3)),
                       ("observation/state", (8,))):
        value = observation[key]
        if not isinstance(value, np.ndarray) or value.shape != shape:
            raise ValueError(f"Invalid {key}: expected array with shape {shape}")
        fingerprints[key] = {"dtype": value.dtype.str, "shape": list(value.shape),
                             "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if not isinstance(observation["prompt"], str):
        raise ValueError("Invalid prompt: expected a string")
    fingerprints["prompt"] = hashlib.sha256(observation["prompt"].encode("utf-8")).hexdigest()
    return fingerprints


def check_round_trip(client, request):
    import numpy as np

    expected = observation_fingerprints(request)
    expected_actions = np.repeat(request["observation/state"][None, :7], 10, axis=0)
    response = client.infer(request)
    received = response.get("observation_fingerprints")
    if not isinstance(received, dict):
        raise ValueError("Missing or invalid observation fingerprints in WebSocket response")
    for key, fingerprint in expected.items():
        if received.get(key) != fingerprint:
            raise ValueError(f"Observation content changed during WebSocket transport: {key}")
    if received.keys() != expected.keys():
        raise ValueError("Unexpected observation fingerprints in WebSocket response")
    np.testing.assert_array_equal(response["actions"], expected_actions)


def serve(port, metadata_file):
    import numpy as np
    from openpi_client.base_policy import BasePolicy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    class EchoStatePolicy(BasePolicy):
        def infer(self, observation):
            fingerprints = observation_fingerprints(observation)
            return {"actions": np.repeat(observation["observation/state"][None, :7], 10, axis=0),
                    "observation_fingerprints": fingerprints}

        def reset(self):
            pass

    WebsocketPolicyServer(EchoStatePolicy(), host="127.0.0.1", port=port,
                          metadata=json.loads(Path(metadata_file).read_text())).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=int)
    parser.add_argument("--metadata")
    parser.add_argument("--output-dir", type=Path, help="Optionally keep reports and mock-server files; otherwise use temporary files")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.server is not None:
        serve(args.server, args.metadata)
        return
    with artifact_directory(args.output_dir) as output:
        report = run_check(output)
        emit_report(report, output / "result.json" if args.output_dir is not None else None)


def run_check(output):
    from pipeline.inference import policy_rpc
    from pipeline.common.config import UPSTREAM_COMMIT
    from pipeline.inference.check_policy import sample_requests
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    run_dir = output / "runs/pi05_libero/transport_test"
    run_dir.mkdir(parents=True)
    model = {"test_only": True}
    norm_sha = "a" * 64
    (run_dir / "config.json").write_text(json.dumps({"model": model, "norm_stats_sha256": norm_sha}))
    metadata = {"pipeline": {"experiment": "transport_test", "checkpoint_step": 119,
                              "upstream_commit": UPSTREAM_COMMIT, "model": model,
                              "norm_stats_sha256": norm_sha}}
    metadata_file = output / "mock_server_metadata.json"
    metadata_file.write_text(json.dumps(metadata))
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
    with (output / "mock_server.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-B", __file__, "--server", str(port),
                                    "--metadata", str(metadata_file)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    received = policy_rpc.read_server_metadata("127.0.0.1", port)
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Mock policy server did not start:\n" +
                                           (output / "mock_server.log").read_text(errors="replace")[-4000:])
                    time.sleep(0.2)
            policy_rpc.verify_server_metadata(received, "transport_test", 119, workspace=output)
            try:
                policy_rpc.verify_server_metadata(received, "transport_test", 99, workspace=output)
            except ValueError:
                pass
            else:
                raise AssertionError("Wrong checkpoint identity was accepted")
            client = WebsocketClientPolicy("127.0.0.1", port)
            count = 0
            for _, _, request in sample_requests([0]):
                check_round_trip(client, request)
                count += 1
            if count != 3:
                raise ValueError(f"Expected 3 real observations, checked {count}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    report = {"status": "PASS", "real_observations_checked": count,
              "wrong_checkpoint_rejected": True,
              "scope": "Official WebSocket server/client and real dataset images with MOCK policy; no checkpoint/model/GPU/simulator"}
    return report


if __name__ == "__main__":
    main()
