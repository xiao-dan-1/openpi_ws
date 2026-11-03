"""Small client-side checks; compatible with both Python 3.8 and 3.11."""
import inspect
import json
import os
from pathlib import Path
import urllib.request

from pipeline.common.runtime import UPSTREAM_COMMIT

WORKSPACE = Path(__file__).resolve().parents[2]


def prepare_client(host):
    # The server is reached directly, including when the shell has HTTP proxies.
    for key in ("NO_PROXY", "no_proxy"):
        values = [value for value in os.environ.get(key, "").split(",") if value]
        for value in ("127.0.0.1", "localhost", host):
            if value not in values:
                values.append(value)
        os.environ[key] = ",".join(values)


def read_server_metadata(host, port):
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect

    prepare_client(host)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{host}:{port}/healthz", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError("Policy server health check failed")
    kwargs = {"compression": None, "max_size": None, "open_timeout": 10}
    if "proxy" in inspect.signature(connect).parameters:
        kwargs["proxy"] = None
    with connect(f"ws://{host}:{port}", **kwargs) as connection:
        return msgpack_numpy.unpackb(connection.recv(timeout=10))


def verify_server_metadata(metadata, experiment, step, *, workspace=None, robot="libero"):
    if robot not in ("libero", "q5"):
        raise ValueError(f"Unsupported robot: {robot}")
    root = WORKSPACE if workspace is None else workspace
    original = json.loads((root / "runs" / f"pi05_{robot}" / experiment / "config.json").read_text())
    identity = metadata.get("pipeline", {})
    expected = {"experiment": experiment, "checkpoint_step": step,
                "upstream_commit": UPSTREAM_COMMIT, "model": original["model"],
                "norm_stats_sha256": original["norm_stats_sha256"]}
    if identity != expected:
        raise ValueError("Connected policy server does not match the requested experiment/checkpoint/stats")
    if robot == "q5":
        from pipeline.q5.schema import JOINT_NAMES, ACTION_REPRESENTATION
        if (metadata.get("robot_type") != "q5" or metadata.get("joint_names") != list(JOINT_NAMES)
                or metadata.get("action_representation") != ACTION_REPRESENTATION):
            raise ValueError("Connected Q5 policy has a different robot, joint order or action representation")
    return identity
