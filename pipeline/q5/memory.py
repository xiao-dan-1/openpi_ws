"""Explicit Q5 JAX pool budget; standard-library only, safe before JAX imports."""
import argparse
import math
import os

DEFAULT_MEMORY_FRACTION = 0.75


def memory_fraction(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("memory-fraction must be a finite number strictly between 0 and 1") from exc
    if not math.isfinite(number) or not 0 < number < 1:
        raise argparse.ArgumentTypeError("memory-fraction must be strictly between 0 and 1")
    return number


def add_memory_argument(parser):
    parser.add_argument("--memory-fraction", type=memory_fraction, default=DEFAULT_MEMORY_FRACTION,
                        help="Per-GPU JAX memory pool fraction of total memory, not free memory (default: 0.75)")


def configure_memory_fraction(value):
    number = memory_fraction(value)
    # Explicit CLI value wins over either spelling of a preexisting fraction;
    # setting both spellings simultaneously makes this JAX version raise.
    os.environ.pop("XLA_CLIENT_MEM_FRACTION", None)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(number)
    return number


def memory_settings():
    return {"memory_fraction": float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", DEFAULT_MEMORY_FRACTION)),
            "preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true") not in ("false", "False", "0"),
            "allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default"),
            "scope": "per-GPU JAX pool budget, not actual live memory or a process-wide hard cap"}
