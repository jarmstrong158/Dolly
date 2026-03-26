"""
Order arrival curve generation.
Distributes daily order volume across time buckets with noise.
"""
import random
import math

from volt_sim.config import (
    ORDER_ARRIVAL_BUCKETS, HIGH_VOLUME_PERCENTILE,
    DAY_START_HOUR, MORNING_END_HOUR, AFTERNOON_END_HOUR,
    STEP_DURATION,
)


def generate_arrival_schedule(total_orders: int, is_high_volume: bool,
                              eod_hour: float) -> dict[float, int]:
    """
    Returns {simulated_hour: num_orders_arriving} for each 30-min step.

    High volume days flatten the curve for even distribution.
    """
    if is_high_volume:
        return _flat_distribution(total_orders, eod_hour)
    return _curved_distribution(total_orders, eod_hour)


def _curved_distribution(total_orders: int, eod_hour: float) -> dict[float, int]:
    # Draw fractions from the configured ranges
    fractions = []
    for _, lo, hi in ORDER_ARRIVAL_BUCKETS:
        fractions.append(random.uniform(lo, hi))

    # Normalize to sum to 1.0
    total_frac = sum(fractions)
    fractions = [f / total_frac for f in fractions]

    # Bucket boundaries in hours
    bucket_boundaries = [
        (DAY_START_HOUR, DAY_START_HOUR),        # instant at day start
        (DAY_START_HOUR, MORNING_END_HOUR),       # morning flow
        (MORNING_END_HOUR, AFTERNOON_END_HOUR),   # afternoon slow
        (AFTERNOON_END_HOUR, eod_hour),            # late surge
    ]

    schedule = {}
    remaining = total_orders

    for i, (frac, (start, end)) in enumerate(zip(fractions, bucket_boundaries)):
        if i == len(fractions) - 1:
            bucket_orders = max(0, remaining)
        else:
            bucket_orders = min(round(total_orders * frac), remaining)
            remaining -= bucket_orders

        if i == 0:
            # Instant arrival at day start
            schedule[DAY_START_HOUR] = bucket_orders
        else:
            # Spread across 30-min steps in this window
            steps = _get_steps_in_range(start, end)
            if not steps:
                # Edge case: add to previous bucket
                schedule[DAY_START_HOUR] = schedule.get(DAY_START_HOUR, 0) + bucket_orders
                continue
            _distribute_to_steps(schedule, steps, bucket_orders)

    return schedule


def _flat_distribution(total_orders: int, eod_hour: float) -> dict[float, int]:
    """Even distribution across all time steps."""
    steps = _get_steps_in_range(DAY_START_HOUR, eod_hour)
    if not steps:
        return {DAY_START_HOUR: total_orders}

    schedule = {}
    _distribute_to_steps(schedule, steps, total_orders)
    return schedule


def _get_steps_in_range(start: float, end: float) -> list[float]:
    steps = []
    t = start
    while t < end - 0.01:
        steps.append(round(t, 2))
        t += STEP_DURATION
    return steps


def _distribute_to_steps(schedule: dict, steps: list[float], num_orders: int):
    """Distribute orders across steps with slight noise."""
    if len(steps) == 0:
        return

    base_per_step = num_orders // len(steps)
    remainder = num_orders - base_per_step * len(steps)

    # Add remainder to random steps
    bonus_steps = set(random.sample(range(len(steps)), min(remainder, len(steps))))

    for i, step in enumerate(steps):
        count = base_per_step + (1 if i in bonus_steps else 0)
        schedule[step] = schedule.get(step, 0) + count


def is_high_volume_day(total_orders: int, month_range: tuple[int, int]) -> bool:
    lo, hi = month_range
    threshold = lo + (hi - lo) * HIGH_VOLUME_PERCENTILE
    return total_orders >= threshold
