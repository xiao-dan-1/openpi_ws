"""Docker launcher for LIBERO simulation entrypoints (pure stdlib, safe under python -S)."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

DEFAULT_IMAGE = "libero:latest"
IN_CONTAINER_ENV = "OPENPI_EVAL_IN_CONTAINER"
CONTAINER_WORKSPACE = Path("/workspace")
CONTAINER_OUTPUT_PARENT = Path("/output_parent")


def in_container():
    return os.environ.get(IN_CONTAINER_ENV) == "1"


def _no_proxy_value(host=None):
    values = ["127.0.0.1", "localhost"]
    if host and host not in values:
        values.append(host)
    for key in ("NO_PROXY", "no_proxy"):
        for item in os.environ.get(key, "").split(","):
            item = item.strip()
            if item and item not in values:
                values.append(item)
    return ",".join(values)


def build_docker_command(*, module, inner_args, workspace, image=DEFAULT_IMAGE,
                         render_backend="egl", host=None, output_dir=None):
    workspace = Path(workspace).resolve()
    args = list(inner_args)
    cmd = [
        "docker", "run", "--rm", "--network", "host",
        "-e", "HOME=/tmp",
        "-e", f"{IN_CONTAINER_ENV}=1",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", f"PYTHONPATH={CONTAINER_WORKSPACE}:{CONTAINER_WORKSPACE / 'openpi/packages/openpi-client/src'}:{CONTAINER_WORKSPACE / 'openpi/third_party/libero'}",
        "-w", str(CONTAINER_WORKSPACE),
        "-v", f"{workspace}:{CONTAINER_WORKSPACE}:ro",
    ]

    no_proxy = _no_proxy_value(host)
    cmd += ["-e", f"NO_PROXY={no_proxy}", "-e", f"no_proxy={no_proxy}"]

    if render_backend == "egl":
        cmd += [
            "--gpus", "all",
            "-e", "MUJOCO_GL=egl",
            "-e", "MUJOCO_EGL_DEVICE_ID=0",
            "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
            "-e", "PYOPENGL_PLATFORM=egl",
        ]
    elif render_backend == "osmesa":
        cmd += [
            "-e", "CUDA_VISIBLE_DEVICES=",
            "-e", "MUJOCO_GL=osmesa",
            "-e", "PYOPENGL_PLATFORM=osmesa",
        ]
    else:
        raise ValueError(f"Unsupported render backend: {render_backend}")

    chown_trap = ""
    if output_dir is not None:
        resolved_output = Path(output_dir).resolve()
        if resolved_output.exists():
            raise FileExistsError(f"Output directory already exists: {resolved_output}")
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        container_output = CONTAINER_OUTPUT_PARENT / resolved_output.name
        cmd += ["-v", f"{resolved_output.parent}:{CONTAINER_OUTPUT_PARENT}"]
        args += ["--output-dir", str(container_output)]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            quoted_out = shlex.quote(str(container_output))
            chown_trap = (
                f"trap 'if [ -e {quoted_out} ]; then "
                f"chown -R {os.getuid()}:{os.getgid()} {quoted_out} 2>/dev/null || true; fi' EXIT; "
            )

    inner_command = " ".join(shlex.quote(part) for part in ["python", "-B", "-u", "-m", module, *args])
    runner = f"source /.venv/bin/activate && {inner_command}" if chown_trap else f"source /.venv/bin/activate && exec {inner_command}"
    cmd += [
        "--entrypoint", "/bin/bash",
        image,
        "-c", f"{chown_trap}{runner}",
    ]
    return cmd


def run_in_docker(*, module, inner_args, workspace, image=DEFAULT_IMAGE,
                  render_backend="egl", host=None, output_dir=None):
    if shutil.which("docker") is None:
        raise RuntimeError("docker executable not found; Docker is required for pipeline.eval")
    cmd = build_docker_command(
        module=module,
        inner_args=inner_args,
        workspace=workspace,
        image=image,
        render_backend=render_backend,
        host=host,
        output_dir=output_dir,
    )
    completed = subprocess.run(cmd, check=False)
    return completed.returncode
