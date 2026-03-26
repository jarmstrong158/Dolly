"""
Episode generator — sets up a complete day scenario.
Rolls date, season, orders, debuffs, side projects.
"""
import random
import calendar
from dataclasses import dataclass

from volt_sim.config import (
    ORDER_VOLUME_RANGES, MONTH_TO_SEASON, PICKER_SCHEDULE,
    RESTOCK_BASE_HOURS, RESTOCK_VOLUME_COEFF, RESTOCK_NOISE_RANGE,
    DELIBERATE_PROJECT_CHANCE, DELIBERATE_PROJECT_SIZE_RANGE,
    FILLER_PROJECT_SIZE, EOD_HOUR, ORDER_CUTOFF_HOUR, DAY_START_HOUR,
)
from volt_sim.env.workers import roll_debuffs, WorkerState
from volt_sim.env.order_arrival import (
    generate_arrival_schedule, is_high_volume_day,
)


@dataclass
class EpisodeConfig:
    # Date info
    year: int
    month: int
    day: int
    day_of_week: int  # 0=Monday
    month_name: str
    season: str

    # Orders
    total_orders: int
    is_high_volume: bool
    arrival_schedule: dict  # {hour: count}

    # Restock
    restock_hours: float

    # Side projects
    has_deliberate_project: bool
    deliberate_project_size: float
    filler_available: bool

    # Workers
    workers: list
    picker_id: int
    picker_needs_replacement: bool  # Blake flare on Tuesday

    # Debuffs summary
    debuffs_fired: list

    # EOD hour (latest shift end across all workers)
    eod_hour: float


def generate_episode(cooldown_tracker: dict = None,
                     force_month: int = None,
                     force_dow: int = None,
                     force_volume: int = None) -> EpisodeConfig:
    if cooldown_tracker is None:
        cooldown_tracker = {}

    # Pick a random working day
    year = random.randint(2024, 2026)
    month = force_month if force_month else random.randint(1, 12)
    month_name = calendar.month_name[month]

    # Pick a valid day-of-week (Mon-Fri) and day-of-month
    if force_dow is not None:
        day_of_week = force_dow
    else:
        day_of_week = random.randint(0, 4)  # Mon-Fri

    # Find a valid day in that month with matching dow
    day = _find_day_for_dow(year, month, day_of_week)

    season = MONTH_TO_SEASON[month]

    # Order volume
    vol_range = ORDER_VOLUME_RANGES[month_name]
    total_orders = force_volume if force_volume else random.randint(vol_range[0], vol_range[1])
    high_vol = is_high_volume_day(total_orders, vol_range)

    # Orders only arrive until ORDER_CUTOFF_HOUR (5:00 PM) — anything after doesn't need completing
    arrival = generate_arrival_schedule(total_orders, high_vol, ORDER_CUTOFF_HOUR)

    # Restock
    noise = random.uniform(*RESTOCK_NOISE_RANGE)
    restock_hours = max(0.0, RESTOCK_BASE_HOURS + total_orders * RESTOCK_VOLUME_COEFF + noise)

    # Side projects
    has_deliberate = random.random() < DELIBERATE_PROJECT_CHANCE
    deliberate_size = 0.0
    if has_deliberate:
        deliberate_size = random.uniform(*DELIBERATE_PROJECT_SIZE_RANGE)

    # Roll debuffs
    workers = roll_debuffs(season, cooldown_tracker)

    # Hustle mode on high-volume days — team pushes harder
    if high_vol:
        for w in workers:
            w.hustle_mode = True

    # Set picker
    picker_id = PICKER_SCHEDULE.get(day_of_week, 4)  # default Reid
    picker_needs_replacement = False

    # Check if Blake is picker (Tuesday) and has flare
    if picker_id == 3:  # Blake
        andrew = workers[3]
        if andrew.is_pack_only:
            picker_needs_replacement = True

    # Mark picker
    if not picker_needs_replacement:
        workers[picker_id].is_picker = True

    # Collect debuffs summary
    debuffs_fired = []
    for w in workers:
        entry = {
            "worker": w.name,
            "sleep": w.sleep_debuff,
            "health": w.health_debuff,
            "individual": w.individual_debuff,
        }
        if w.sleep_debuff != "normal" or w.health_debuff != "normal" or w.individual_debuff:
            debuffs_fired.append(entry)

    return EpisodeConfig(
        year=year,
        month=month,
        day=day,
        day_of_week=day_of_week,
        month_name=month_name,
        season=season,
        total_orders=total_orders,
        is_high_volume=high_vol,
        arrival_schedule=arrival,
        restock_hours=restock_hours,
        has_deliberate_project=has_deliberate,
        deliberate_project_size=deliberate_size,
        filler_available=True,
        workers=workers,
        picker_id=picker_id,
        picker_needs_replacement=picker_needs_replacement,
        debuffs_fired=debuffs_fired,
        eod_hour=EOD_HOUR,
    )


def _find_day_for_dow(year: int, month: int, target_dow: int) -> int:
    """Find the first day in month/year matching target day-of-week."""
    for day in range(1, 29):  # safe for all months
        try:
            dow = calendar.weekday(year, month, day)
            if dow == target_dow:
                return day
        except ValueError:
            continue
    return 1
