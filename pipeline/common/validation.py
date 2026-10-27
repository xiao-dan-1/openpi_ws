"""Argument validation shared by training, inference, and evaluation."""
import argparse


def experiment_name(value):
    if not value or value in {".", ".."} or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value
    ):
        raise argparse.ArgumentTypeError(
            "Use an experiment name containing only letters, numbers, '-' and '_'"
        )
    return value
