"""Argument validation shared by training, inference, and evaluation."""
import argparse


def require(condition, message):
    if not condition:
        raise ValueError(message)


def gpu_selectors(value):
    parts = value.split(",")
    if len(parts) not in (1, 2) or len(parts) != len(set(parts)) or not all(
        p.isdigit() or (p.startswith("GPU-") and len(p) > 4 and p.strip() == p) for p in parts
    ):
        raise argparse.ArgumentTypeError("Use one GPU or two distinct GPUs, e.g. 2 or 2,3 (UUIDs also accepted)")
    canonical = [str(int(p)) if p.isdigit() else p for p in parts]
    if len(set(canonical)) != len(parts):
        raise argparse.ArgumentTypeError("GPU selectors must refer to distinct devices")
    return ",".join(canonical)


def experiment_name(value):
    if not value or value in {".", ".."} or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value
    ):
        raise argparse.ArgumentTypeError(
            "Use an experiment name containing only letters, numbers, '-' and '_'"
        )
    return value
