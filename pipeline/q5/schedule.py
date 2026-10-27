"""Explicit segment/schedule semantics, independent from JAX or model imports."""


def validate_schedule(profile, segment_steps, warmup_steps, schedule_steps, continue_after_segment=False):
    if segment_steps < 1 or warmup_steps < 0 or schedule_steps <= warmup_steps:
        raise ValueError("Require positive segment, nonnegative warmup, and schedule_steps > warmup_steps")
    if segment_steps > schedule_steps:
        raise ValueError("Segment limit exceeds the declared schedule; explicitly extend the schedule first")
    if profile == "finetune" and segment_steps <= warmup_steps and not continue_after_segment:
        raise ValueError("This finetune segment ends during warmup. Add --continue-after-segment for an intended continuation, or explicitly shorten --warmup-steps for a standalone run")
    return {"segment_limit": segment_steps, "warmup_steps": warmup_steps, "schedule_steps": schedule_steps,
            "continue_after_segment": continue_after_segment,
            "segment_entirely_in_warmup": segment_steps <= warmup_steps,
            "schedule_endpoint_reached_by_segment": segment_steps >= schedule_steps}
