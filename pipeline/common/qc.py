"""Converter QC fields shared by reports and training admission checks."""
import math

COUNTERS = ("no_state", "no_action", "stale_state", "stale_action", "disorder", "age_a_future")
MAXIMA = ("age_s_max", "age_a_max", "back_max")
MINIMA = ("age_s_min", "age_a_min")


def fields(qc):
    """Missing measurements are unknown; never manufacture zero counters."""
    return {key: qc.get(key) for key in (*COUNTERS, *MAXIMA, *MINIMA)}


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def require_training_qc(qc):
    """Require measured QC fields for the CPU training-data contract check."""
    values = fields(qc)
    for key in (*COUNTERS, "age_s_max", "age_a_max"):
        value = values[key]
        if not is_number(value) or value < 0 or (key in COUNTERS and not float(value).is_integer()):
            raise ValueError(f"Missing or invalid converter QC field: {key}")
    return values
