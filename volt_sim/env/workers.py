"""
Worker state management and debuff system.
"""
import random
from dataclasses import dataclass, field
from typing import Optional

from volt_sim.config import (
    WORKERS, SLEEP_DEBUFFS, HEALTH_DEBUFFS, BAD_HEADSPACE_EFFECTS,
    INDIVIDUAL_DEBUFFS, BLAKE_FLARE_PROBABILITIES, FATIGUE_OPH_PENALTY,
    PACK_FATIGUE_THRESHOLD, PICK_FATIGUE_THRESHOLD,
    TRENT_SORENESS_HOUR_THRESHOLD, TRENT_SORENESS_OPH_PENALTY,
    DAY_START_HOUR, TASKS, TASK_TO_IDX,
    HUSTLE_MODE_BONUS,
)


@dataclass
class WorkerState:
    worker_id: int
    name: str
    base_oph: float
    shift_hours: float
    role: str

    # Current state
    current_task: str = "idle"
    hours_worked: float = 0.0
    shift_start: float = DAY_START_HOUR
    is_picker: bool = False

    # Debuffs
    sleep_debuff: str = "normal"
    sleep_multiplier: float = 1.0
    health_debuff: str = "normal"
    health_multiplier: float = 1.0
    individual_debuff: Optional[str] = None
    individual_multiplier: float = 1.0
    individual_effect: Optional[str] = None

    # Individual debuff specifics
    hours_lost: float = 0.0       # for family_needs
    is_absent: bool = False        # for Reid's NCNS
    is_pack_only: bool = False     # for Blake's flare
    has_soreness: bool = False     # for Trent

    # Fatigue tracking
    orders_packed: int = 0
    orders_picked: int = 0
    is_fatigued: bool = False

    # Fractional work accumulator — avoids int truncation waste on short steps
    work_carry: float = 0.0

    # Trent soreness tracking
    non_side_project_hours: float = 0.0
    soreness_activated: bool = False

    # Marcus management
    management_hours: float = 0.0

    # Hustle mode — high-volume day push
    hustle_mode: bool = False

    @property
    def shift_end(self) -> float:
        return self.shift_start + self.shift_hours

    @property
    def hours_remaining(self) -> float:
        return max(0.0, self.shift_hours - self.hours_worked - self.hours_lost)

    def effective_oph(self, task: str = None) -> float:
        if task is None:
            task = self.current_task

        base = self.base_oph
        sleep_mod = self.sleep_multiplier
        health_mod = self._health_modifier_for_task(task)
        individual_mod = self.individual_multiplier
        fatigue_mod = self._fatigue_modifier()

        hustle_mod = HUSTLE_MODE_BONUS if self.hustle_mode else 1.0

        return base * sleep_mod * health_mod * individual_mod * fatigue_mod * hustle_mod

    def _health_modifier_for_task(self, task: str) -> float:
        if self.health_debuff != "bad_headspace":
            return self.health_multiplier

        effects = BAD_HEADSPACE_EFFECTS.get(self.name, {"default": 1.0})
        if task in effects:
            return effects[task]
        return effects.get("default", 1.0)

    def _fatigue_modifier(self) -> float:
        mod = 1.0
        if self.is_fatigued:
            mod *= FATIGUE_OPH_PENALTY
        if self.has_soreness and self.soreness_activated:
            mod *= TRENT_SORENESS_OPH_PENALTY
        return mod

    def check_fatigue(self):
        if not self.is_fatigued:
            if self.orders_packed >= PACK_FATIGUE_THRESHOLD or \
               self.orders_picked >= PICK_FATIGUE_THRESHOLD:
                self.is_fatigued = True

    def check_trent_soreness(self):
        if self.has_soreness and not self.soreness_activated:
            if self.non_side_project_hours >= TRENT_SORENESS_HOUR_THRESHOLD:
                self.soreness_activated = True

    def can_do_task(self, task: str) -> bool:
        if self.is_absent:
            return False
        if self.is_pack_only and task not in ("pack", "idle"):
            return False
        if self.hours_remaining <= 0:
            return task == "idle"
        return True


def _roll_category(debuff_list: list[dict]) -> dict:
    roll = random.random()
    cumulative = 0.0
    for debuff in debuff_list:
        cumulative += debuff["probability"]
        if roll < cumulative:
            return debuff
    return debuff_list[-1]


def roll_debuffs(season: str, cooldown_tracker: dict) -> list[WorkerState]:
    workers = []

    for cfg in WORKERS:
        w = WorkerState(
            worker_id=cfg["id"],
            name=cfg["name"],
            base_oph=cfg["oph"],
            shift_hours=cfg["shift_hours"],
            role=cfg["role"],
        )

        # Roll sleep category
        sleep = _roll_category(SLEEP_DEBUFFS)
        w.sleep_debuff = sleep["name"]
        w.sleep_multiplier = sleep["multiplier"]

        # Roll health category
        health = _roll_category(HEALTH_DEBUFFS)
        w.health_debuff = health["name"]
        if health["name"] == "bad_headspace":
            w.health_multiplier = 1.0  # handled per-task
        elif health["name"] == "buffer":
            w.health_debuff = "normal"
            w.health_multiplier = 1.0
        else:
            w.health_multiplier = health["multiplier"]

        # Roll individual debuff
        indiv_cfg = INDIVIDUAL_DEBUFFS.get(w.name)
        if indiv_cfg:
            _roll_individual_debuff(w, indiv_cfg, season, cooldown_tracker)

        workers.append(w)

    return workers


def _roll_individual_debuff(w: WorkerState, cfg: dict, season: str,
                            cooldown_tracker: dict):
    key = w.name
    cooldown_remaining = cooldown_tracker.get(key, 0)

    if cooldown_remaining > 0:
        cooldown_tracker[key] = cooldown_remaining - 1
        return

    # Determine probability
    if cfg["name"] == "eoe_muscle_flare":
        prob = BLAKE_FLARE_PROBABILITIES.get(season, 0.10)
    else:
        prob = cfg["probability"]

    if random.random() >= prob:
        return

    # Debuff fires
    w.individual_debuff = cfg["name"]

    if cfg["cooldown_days"] > 0:
        cooldown_tracker[key] = cfg["cooldown_days"]

    effect = cfg["effect"]
    if effect == "lose_hours":
        lo, hi = cfg["hours_lost_range"]
        w.hours_lost = random.uniform(lo, hi)
        w.individual_multiplier = 1.0
        w.individual_effect = "lose_hours"

    elif effect == "oph_penalty":
        w.individual_multiplier = cfg["oph_multiplier"]
        w.individual_effect = "oph_penalty"

    elif effect == "pack_only":
        w.is_pack_only = True
        w.individual_multiplier = 1.0
        w.individual_effect = "pack_only"

    elif effect == "absent":
        w.is_absent = True
        w.individual_multiplier = 0.0
        w.individual_effect = "absent"

    elif effect == "soreness":
        w.has_soreness = True
        w.individual_multiplier = 1.0
        w.individual_effect = "soreness"
