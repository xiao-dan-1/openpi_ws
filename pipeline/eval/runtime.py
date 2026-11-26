"""Select the local LIBERO environment or an explicitly requested Docker image."""
import os
from pathlib import Path
import sys
import sysconfig

from pipeline.eval.docker_runtime import DEFAULT_IMAGE, in_container, run_in_docker


def add_runtime_arguments(parser):
    parser.add_argument("--render-backend", choices=["osmesa", "egl"],
                        help="MuJoCo rendering backend (default: osmesa locally, egl in Docker)")
    parser.add_argument("--docker", action="store_true", help="Run the simulator in Docker")
    parser.add_argument("--image", help=f"Select a Docker image (implies --docker, default: {DEFAULT_IMAGE})")


def dispatch_simulation(parser, args, *, module, workspace, docker_args, host=None):
    """Return whether Docker has already run the simulation."""
    container = in_container()
    docker_requested = args.docker or args.image is not None
    if args.render_backend is None:
        args.render_backend = "egl" if container or docker_requested else "osmesa"
    if container:
        return False
    if docker_requested:
        code = run_in_docker(
            module=module,
            inner_args=[*docker_args, "--render-backend", args.render_backend],
            workspace=workspace,
            image=args.image if args.image is not None else DEFAULT_IMAGE,
            render_backend=args.render_backend,
            host=host,
            output_dir=args.output_dir,
        )
        if code:
            raise SystemExit(code)
        return True
    _ensure_local_runtime(parser, module, workspace, args.render_backend)
    return False


def _ensure_local_runtime(parser, module, workspace, backend):
    current_python = Path(sys.executable)
    local_python = workspace / "tools/libero-env/bin/python"
    python = current_python
    python38 = sys.version_info[:2] == (3, 8)
    if ((not python38 or sys.prefix == sys.base_prefix)
            and local_python.is_file() and os.access(local_python, os.X_OK)):
        python = local_python
    if python == current_python and not python38:
        parser.error("LIBERO requires Python 3.8. Run: bash scripts/setup_libero_env.sh")

    env = os.environ.copy()
    original_library_path = env.get("LD_LIBRARY_PATH")
    if backend == "osmesa":
        multiarch = sysconfig.get_config_var("MULTIARCH") or ""
        library_dir = workspace / "tools/osmesa/usr/lib" / multiarch
        if (library_dir / "libOSMesa.so.8").is_file():
            paths = original_library_path.split(os.pathsep) if original_library_path else []
            if str(library_dir) not in paths:
                env["LD_LIBRARY_PATH"] = os.pathsep.join([*paths, str(library_dir)])

    if python != current_python or env.get("LD_LIBRARY_PATH") != original_library_path:
        # The dynamic loader reads LD_LIBRARY_PATH at process startup.
        try:
            os.execve(python, [str(python), "-B", "-u", "-m", module, *sys.argv[1:]], env)
        except OSError as error:
            parser.error(f"Could not start LIBERO with {python}: {error}")
