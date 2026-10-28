"""Optional report persistence; compatible with the Python 3.8 simulator."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile


def emit_report(report, output=None):
    text = json.dumps(report, indent=2)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


@contextmanager
def artifact_directory(output_dir):
    """Keep explicitly requested artifacts; otherwise clean up temporary I/O."""
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="openpi-check-") as temporary:
            yield Path(temporary)
    else:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True)
        yield output
