"""
Warehouse environment — the core simulation.
Steps through a simulated work day in 15-minute intervals.
"""
import random
import numpy as np
from typing import Optional

from volt_sim.config import (
    NUM_WORKERS, NUM_TASKS, TASKS, TASK_TO_IDX,
    DAY_START_HOUR, LUNCH_HOUR, LUNCH_DURATION, EOD_HOUR,
    STEP_DURATION, TOTAL_STATE_SIZE,
    WORKER_STATE_SIZE, ENV_STATE_SIZE, SEASONS,
    MARCUS_MANAGEMENT_HOURS_REQUIRED, REWARDS, OT_WALL_CLOCK_MAX, OT_HARD_STOP,
    PACK_FATIGUE_THRESHOLD, PICK_FATIGUE_THRESHOLD,
    TASK_OPH_MULTIPLIER, PICK_MULTIPLIER_MAIN, PICK_MULTIPLIER_SUPPLEMENT,
    MORNING_PICK_CARTS_MIN, MORNING_PICK_CARTS_MAX,
    MORNING_PICK_PER_CART_MIN, MORNING_PICK_PER_CART_MAX,
    FILLER_COMPLETION_THRESHOLD,
    MARCUS_PRE_SIM_MANAGEMENT,
)
from volt_sim.env.episode_generator import generate_episode, EpisodeConfig
from volt_sim.env.workers import WorkerState


class WarehouseEnv:
    def __init__(self):
        self.episode: Optional[EpisodeConfig] = None
        self.current_hour: float = DAY_START_HOUR
        self.cooldown_tracker: dict = {}

        # Order tracking
        self.orders_in_queue: int = 0
        self.orders_completed: int = 0
        self.orders_picked_not_audited: int = 0

        # Restock tracking
        self.restock_remaining: float = 0.0

        # Side project tracking
        self.deliberate_progress: float = 0.0
        self.filler_progress: float = 0.0
        self.deliberate_complete: bool = False
        self.filler_complete: bool = False

        # Reward tracking
        self.total_reward: float = 0.0
        self.reward_breakdown: dict = {}
        self.ot_hours: float = 0.0
        self.is_ot: bool = False
        self.is_done: bool = False

        # Per-worker order counters for Marcus/Nolan penalty
        self.marcus_orders: int = 0
        self.nolan_orders: int = 0

        # Restock interruption tracking
        self.restock_interrupted_pick: bool = False

        # Step log for episode logging
        self.step_log: list = []

    def reset(self, force_month: int = None, force_dow: int = None,
              force_volume: int = None) -> np.ndarray:
        self.episode = generate_episode(
            cooldown_tracker=self.cooldown_tracker,
            force_month=force_month,
            force_dow=force_dow,
            force_volume=force_volume,
        )
        self.current_hour = DAY_START_HOUR
        self.orders_in_queue = 0
        self.orders_completed = 0
        self.orders_picked_not_audited = 0
        self.restock_remaining = self.episode.restock_hours
        self.deliberate_progress = 0.0
        self.filler_progress = 0.0
        self.deliberate_complete = False
        self.filler_complete = False
        self.total_reward = 0.0
        self.reward_breakdown = {k: 0.0 for k in REWARDS}
        self.ot_hours = 0.0
        self.is_ot = False
        self.is_done = False
        self.marcus_orders = 0
        self.nolan_orders = 0
        self.restock_interrupted_pick = False
        self.step_log = []

        # Process initial order arrivals
        self._process_arrivals()

        # Marcus's pre-sim management (7:45-9:00)
        self.episode.workers[0].management_hours = MARCUS_PRE_SIM_MANAGEMENT

        # Morning pick round: everyone grabs 1-2 carts before assignments
        self._morning_pick_round()

        # Default assignments: designated picker picks, everyone else packs.
        # The agent's 2 reassignments per step handle the flex (Marcus/Nolan floating).
        for w in self.episode.workers:
            if w.is_absent:
                w.current_task = "idle"
            elif w.is_picker:
                w.current_task = "pick"
            elif w.worker_id == 0:  # Marcus — management first
                w.current_task = "management"
            else:
                w.current_task = "pack"

        return self._get_state()

    def _morning_pick_round(self):
        """All available workers pick 1-2 carts at day start. Fixed mechanic."""
        for w in self.episode.workers:
            if w.is_absent:
                continue
            num_carts = random.randint(MORNING_PICK_CARTS_MIN, MORNING_PICK_CARTS_MAX)
            cart_total = sum(
                random.randint(MORNING_PICK_PER_CART_MIN, MORNING_PICK_PER_CART_MAX)
                for _ in range(num_carts)
            )
            picked = min(cart_total, self.orders_in_queue)
            if picked > 0:
                self.orders_in_queue -= picked
                self.orders_picked_not_audited += picked
                w.orders_picked += picked

    def step(self, actions: list[tuple[int, int]]) -> tuple[np.ndarray, float, bool, dict]:
        """
        actions: list of (worker_id, task_idx) tuples — one per worker.
        Every step is a full assignment — agent controls all workers every 15 min.
        Returns: (state, reward, done, info)
        """
        step_reward = 0.0

        for worker_id, task_idx in actions:
            if 0 <= worker_id < NUM_WORKERS and 0 <= task_idx < NUM_TASKS:
                worker = self.episode.workers[worker_id]
                new_task = TASKS[task_idx]

                # Enforce Blake pack-only constraint
                if worker.is_pack_only and new_task not in ("pack", "idle"):
                    step_reward += self._add_reward("blake_prohibited_task")
                    continue

                if worker.can_do_task(new_task):
                    if worker.current_task != new_task:
                        worker.work_carry = 0.0  # reset carry on task switch
                    worker.current_task = new_task

        # Check lunch
        is_lunch = (abs(self.current_hour - LUNCH_HOUR) < 0.01)
        if is_lunch:
            # Process arrivals that happen during lunch (13.0 and 13.25)
            self._process_arrivals()  # 13.0
            self.current_hour += STEP_DURATION
            self._process_arrivals()  # 13.25 (orders arrive even during lunch)
            self.current_hour += STEP_DURATION  # now at 13.5

            # All workers stop for lunch
            for w in self.episode.workers:
                w.current_task = "idle"
            self._log_step(step_reward)
            self.total_reward += step_reward
            # After lunch, return state for reassignment
            return self._get_state(), step_reward, False, self._get_info()

        # Process arrivals BEFORE work so orders available this step get worked
        self._process_arrivals()

        # Simulate work for this 30-min step
        step_reward += self._simulate_step()

        # Advance time
        self.current_hour += STEP_DURATION

        # Check if day is over — finalization rewards added to step_reward
        finalize_reward = self._check_eod()
        step_reward += finalize_reward

        self._log_step(step_reward)
        self.total_reward += step_reward

        return self._get_state(), step_reward, self.is_done, self._get_info()

    def _simulate_step(self) -> float:
        reward = 0.0
        duration = STEP_DURATION  # 0.25 hours (15 min)

        # Categorize workers by availability and task
        active_workers = []
        for w in self.episode.workers:
            if w.is_absent:
                if w.current_task != "idle":
                    w.current_task = "idle"
                continue
            if self.is_ot:
                # During OT everyone stays to finish orders
                active_workers.append(w)
                continue
            if w.hours_remaining <= 0:
                w.current_task = "idle"
                continue
            worker_eod = w.shift_start + w.shift_hours - w.hours_lost
            if self.current_hour >= worker_eod:
                w.current_task = "idle"
                continue
            active_workers.append(w)

        # --- Phase 1: Process pickers first (fills picked_not_audited) ---
        for w in active_workers:
            if w.current_task != "pick":
                continue
            oph = w.effective_oph("pick")
            # Designated picker gets higher multiplier than supplemental pickers
            pick_mult = PICK_MULTIPLIER_MAIN if w.is_picker else PICK_MULTIPLIER_SUPPLEMENT
            raw_output = oph * pick_mult * duration + w.work_carry
            picked = min(int(raw_output), self.orders_in_queue)
            # Only carry fractional remainder from actual work, not idle accumulation
            if self.orders_in_queue > 0:
                w.work_carry = raw_output - picked
            else:
                w.work_carry = 0.0
            self.orders_in_queue -= picked
            self.orders_picked_not_audited += picked
            w.orders_picked += picked
            w.check_fatigue()
            w.hours_worked += duration
            reward += self._add_reward("per_productive_hour", duration)

            # Trent soreness check
            if w.has_soreness:
                w.non_side_project_hours += duration
                w.check_trent_soreness()

            # Restock interruption check
            if self.restock_remaining > 2.0 and self.orders_in_queue > 0:
                if not self.restock_interrupted_pick:
                    self.restock_interrupted_pick = True
                    reward += self._add_reward("restock_pick_interruption")

        # --- Phase 2: Distribute available picked orders fairly to packers ---
        packers = [w for w in active_workers if w.current_task == "pack"]
        if packers and self.orders_picked_not_audited > 0:
            # Calculate each packer's raw output and capacity
            packer_raws = []
            packer_capacities = []
            for w in packers:
                oph = w.effective_oph("pack")
                task_mult = TASK_OPH_MULTIPLIER.get("pack", 1.0)
                raw = oph * task_mult * duration + w.work_carry
                capacity = int(raw)
                packer_raws.append(raw)
                packer_capacities.append(capacity)

            total_capacity = sum(packer_capacities)
            available = self.orders_picked_not_audited

            for i, w in enumerate(packers):
                if available <= 0:
                    share = 0
                elif total_capacity <= available:
                    # Enough orders for everyone — each packer gets full capacity
                    share = packer_capacities[i]
                else:
                    # Not enough orders — distribute proportionally
                    proportion = packer_capacities[i] / max(1, total_capacity)
                    share = min(packer_capacities[i], int(available * proportion))
                    # Give remainder to last packer
                    if i == len(packers) - 1:
                        share = min(packer_capacities[i], available)

                actual = min(share, self.orders_picked_not_audited)
                # Carry only the fractional part (< 1.0) to prevent burst spikes
                if actual > 0:
                    w.work_carry = min(packer_raws[i] - actual, 1.0)
                else:
                    w.work_carry = 0.0
                self.orders_picked_not_audited -= actual
                self.orders_completed += actual
                w.orders_packed += actual
                available -= actual
                w.check_fatigue()

                reward += self._add_reward("per_order_shipped", actual)

                # Track Marcus/Nolan order penalties
                if w.worker_id == 0:
                    self.marcus_orders += actual
                    reward += self._add_reward("marcus_per_order", actual)
                elif w.worker_id == 1:
                    self.nolan_orders += actual
                    reward += self._add_reward("nolan_per_order", actual)

        # Penalty: packers assigned but nothing to pack while queue has orders
        # This teaches the agent that packing without picking is wasted labor
        if packers and self.orders_picked_not_audited == 0 and self.orders_in_queue > 0:
            starved_count = len(packers)
            reward += self._add_reward("packers_starved", starved_count)

        # Penalty: picked orders piling up without being packed — need more packers
        if self.orders_picked_not_audited > 20:
            backlog_units = self.orders_picked_not_audited // 10
            reward += self._add_reward("picked_backlog", backlog_units)

        # Update packer work hours and tracking
        for w in packers:
            w.hours_worked += duration
            reward += self._add_reward("per_productive_hour", duration)
            if w.has_soreness:
                w.non_side_project_hours += duration
                w.check_trent_soreness()

        # --- Phase 3: Process restock, side projects, management, idle ---
        for w in active_workers:
            task = w.current_task
            if task in ("pick", "pack"):
                continue  # already handled

            if task == "idle":
                reward += self._add_reward("per_idle_hour", duration)
                continue

            if task == "management":
                total_mgmt = sum(mw.management_hours for mw in self.episode.workers
                                 if mw.worker_id in (0, 1))
                if total_mgmt < MARCUS_MANAGEMENT_HOURS_REQUIRED:
                    # Still has management duty to fill
                    w.management_hours += duration
                    w.hours_worked += duration
                    reward += self._add_reward("per_productive_hour", duration)
                else:
                    # Quota already met — this is wasted time, treat as idle
                    reward += self._add_reward("per_idle_hour", duration)
                    w.hours_worked += duration
                continue

            # Scale work output by worker effectiveness (debuffs reduce output)
            effectiveness = w.effective_oph(task) / max(1.0, w.base_oph)
            effective_duration = duration * effectiveness

            if task == "restock":
                prev_remaining = self.restock_remaining
                self.restock_remaining = max(0.0, self.restock_remaining - effective_duration)
                # Only fire completion reward once — when restock transitions to 0
                if prev_remaining > 0 and self.restock_remaining <= 0:
                    reward += self._add_reward("per_restock_completed", 1)
                # Penalize using warehouse workers on restock — Marcus/Nolan preferred
                if w.worker_id not in (0, 1):
                    reward += self._add_reward("warehouse_worker_restock")

            elif task == "side_project":
                # Check crunch condition
                if self.orders_in_queue + self.orders_picked_not_audited > self.episode.total_orders * 0.3:
                    reward += self._add_reward("side_project_during_crunch")

                if self.episode.has_deliberate_project and not self.deliberate_complete:
                    self.deliberate_progress += effective_duration
                    reward += self._add_reward("per_deliberate_unit", effective_duration)
                    if self.deliberate_progress >= self.episode.deliberate_project_size:
                        self.deliberate_complete = True
                        reward += self._add_reward("deliberate_completion_bonus")
                else:
                    self.filler_progress += effective_duration
                    reward += self._add_reward("per_filler_unit", effective_duration)

            w.hours_worked += duration
            reward += self._add_reward("per_productive_hour", duration)

            # Trent soreness check
            if w.has_soreness and task != "side_project":
                w.non_side_project_hours += duration
                w.check_trent_soreness()

        return reward

    def _process_arrivals(self):
        if self.episode is None:
            return
        key = round(self.current_hour, 2)
        arrivals = self.episode.arrival_schedule.pop(key, 0)  # pop so we don't double-count
        self.orders_in_queue += arrivals

    def _check_eod(self) -> float:
        """Returns finalization reward (0.0 if day isn't over yet)."""
        orders_remaining = self.orders_in_queue + self.orders_picked_not_audited

        if self.current_hour >= EOD_HOUR:
            if orders_remaining > 0:
                if not self.is_ot:
                    self.is_ot = True
                self.ot_hours = self.current_hour - EOD_HOUR
                # Hard stop at 6:30 PM — no one works past this
                if self.current_hour >= OT_HARD_STOP:
                    return self._finalize_episode()
            else:
                return self._finalize_episode()
        return 0.0

    def _finalize_episode(self) -> float:
        """End-of-day scoring. Returns total finalization reward."""
        self.is_done = True
        reward = 0.0

        orders_remaining = self.orders_in_queue + self.orders_picked_not_audited

        # OT penalty
        if self.ot_hours > 0:
            reward += self._add_reward("per_ot_hour", self.ot_hours)

        # Order completion
        if orders_remaining <= 0:
            reward += self._add_reward("all_orders_complete_bonus")
        else:
            # -10 per incomplete order (single penalty, no double-counting)
            reward += self._add_reward("per_order_incomplete", orders_remaining)
            # Extra flat penalty if OT was used and STILL couldn't finish
            if self.ot_hours > 0:
                reward += self._add_reward("ot_incomplete_flat")

        # Restock EOD check
        if self.restock_remaining <= 0:
            reward += self._add_reward("all_restock_bonus")
        else:
            restock_tasks_remaining = max(1, int(self.restock_remaining / STEP_DURATION))
            reward += self._add_reward("per_restock_bleed", restock_tasks_remaining)

        # Management check — shared between Marcus & Nolan, waived on 400+ days
        total_mgmt = sum(w.management_hours for w in self.episode.workers if w.worker_id in (0, 1))
        if total_mgmt >= MARCUS_MANAGEMENT_HOURS_REQUIRED:
            reward += self._add_reward("management_duty_met")
        elif self.episode.total_orders >= 400:
            pass  # heavy day, management backlog is acceptable
        else:
            reward += self._add_reward("management_duty_missed")

        # Filler completion check
        if self.filler_progress >= FILLER_COMPLETION_THRESHOLD:
            self.filler_complete = True
            reward += self._add_reward("filler_completion_bonus")

        return reward

    def _add_reward(self, key: str, multiplier: float = 1.0) -> float:
        value = REWARDS[key] * multiplier
        self.reward_breakdown[key] = self.reward_breakdown.get(key, 0.0) + value
        return value

    def _get_state(self) -> np.ndarray:
        state = np.zeros(TOTAL_STATE_SIZE, dtype=np.float32)
        idx = 0

        for w in self.episode.workers:
            # One-hot task encoding
            task_idx = TASK_TO_IDX.get(w.current_task, 4)  # default idle
            state[idx + task_idx] = 1.0
            idx += NUM_TASKS

            # Scalar features
            state[idx] = w.effective_oph() / 25.0  # normalize
            idx += 1
            state[idx] = w.hours_worked / 10.0
            idx += 1
            state[idx] = w.hours_remaining / 10.0
            idx += 1
            state[idx] = 1.0 if (w.sleep_debuff != "normal" or w.health_debuff != "normal") else 0.0
            idx += 1
            state[idx] = 1.0 if w.individual_debuff is not None else 0.0
            idx += 1
            state[idx] = 1.0 if w.is_fatigued else 0.0
            idx += 1
            state[idx] = 1.0 if w.is_picker else 0.0
            idx += 1
            state[idx] = 1.0 if w.is_pack_only else 0.0  # Blake flare
            idx += 1
            # Trent's soreness progress (0-1, hits threshold at 1.0)
            from volt_sim.config import TRENT_SORENESS_HOUR_THRESHOLD
            state[idx] = w.non_side_project_hours / TRENT_SORENESS_HOUR_THRESHOLD if w.has_soreness else 0.0
            idx += 1
            state[idx] = w.management_hours / max(1.0, MARCUS_MANAGEMENT_HOURS_REQUIRED)
            idx += 1

        # Environment features
        state[idx] = (self.current_hour - DAY_START_HOUR) / (EOD_HOUR - DAY_START_HOUR)
        idx += 1
        state[idx] = self.orders_in_queue / max(1, self.episode.total_orders)
        idx += 1
        state[idx] = self.orders_completed / max(1, self.episode.total_orders)
        idx += 1
        state[idx] = self.orders_picked_not_audited / max(1, self.episode.total_orders)
        idx += 1
        state[idx] = self.restock_remaining / max(1.0, self.episode.restock_hours)
        idx += 1

        # Side project progress
        if self.episode.has_deliberate_project and self.episode.deliberate_project_size > 0:
            state[idx] = self.deliberate_progress / self.episode.deliberate_project_size
        else:
            state[idx] = self.filler_progress / 10.0
        idx += 1

        # Season one-hot
        season_idx = SEASONS.index(self.episode.season)
        state[idx + season_idx] = 1.0
        idx += 4

        state[idx] = 1.0 if self.episode.is_high_volume else 0.0
        idx += 1

        total_mgmt = sum(w.management_hours for w in self.episode.workers if w.worker_id in (0, 1))
        state[idx] = total_mgmt / MARCUS_MANAGEMENT_HOURS_REQUIRED
        idx += 1

        state[idx] = 1.0 if self.episode.picker_needs_replacement else 0.0
        idx += 1

        return state

    def _get_info(self) -> dict:
        return {
            "current_hour": self.current_hour,
            "orders_in_queue": self.orders_in_queue,
            "orders_completed": self.orders_completed,
            "orders_picked_not_audited": self.orders_picked_not_audited,
            "total_orders": self.episode.total_orders,
            "restock_remaining": self.restock_remaining,
            "total_reward": self.total_reward,
            "is_ot": self.is_ot,
            "picker_needs_replacement": self.episode.picker_needs_replacement,
        }

    def _log_step(self, step_reward: float):
        workers_state = []
        for w in self.episode.workers:
            # Distinguish management from idle in logs
            logged_task = w.current_task
            total_mgmt = sum(mw.management_hours for mw in self.episode.workers if mw.worker_id in (0, 1))
            if logged_task == "management" and total_mgmt >= MARCUS_MANAGEMENT_HOURS_REQUIRED:
                logged_task = "idle"  # quota met, over-management = wasted time
            workers_state.append({
                "name": w.name,
                "task": logged_task,
                "effective_oph": round(w.effective_oph(), 2),
                "debuffs": {
                    "sleep": w.sleep_debuff,
                    "health": w.health_debuff,
                    "individual": w.individual_debuff,
                },
            })

        self.step_log.append({
            "hour": round(self.current_hour, 2),
            "workers": workers_state,
            "orders_remaining": self.orders_in_queue,
            "orders_completed": self.orders_completed,
            "picked_not_audited": self.orders_picked_not_audited,
            "restock_remaining": round(self.restock_remaining, 2),
            "side_project_progress": round(
                self.deliberate_progress if self.episode.has_deliberate_project
                else self.filler_progress, 2
            ),
            "running_reward": round(self.total_reward, 2),
        })

    def get_episode_summary(self) -> dict:
        ep = self.episode
        total = ep.total_orders
        shipped = self.orders_completed
        remaining = self.orders_in_queue + self.orders_picked_not_audited

        # Outcome-based grading — simple and strict
        # Not completing all orders is always an F.
        # A = all orders + restock + management duty + no OT
        # Each breach drops one letter grade.
        all_orders = (shipped >= total)
        restock_pct_raw = 1.0 - (self.restock_remaining / max(0.01, ep.restock_hours))
        all_restock = (restock_pct_raw >= 0.95)

        total_mgmt_grade = sum(w.management_hours for w in ep.workers if w.worker_id in (0, 1))
        if total >= 400:
            mgmt_met = True  # waived on heavy days
        else:
            mgmt_met = (total_mgmt_grade >= MARCUS_MANAGEMENT_HOURS_REQUIRED)

        no_ot = (self.ot_hours <= 0)

        if not all_orders:
            grade = "F"
        else:
            # Start at A, drop one letter per breach
            demerits = 0
            if not all_restock:
                demerits += 1
            if not mgmt_met:
                demerits += 1
            if not no_ot:
                demerits += 1
            grades = ["A", "B", "C", "D", "F"]
            grade = grades[min(demerits, 4)]

        restock_pct = 1.0 - (self.restock_remaining / max(0.01, ep.restock_hours))

        return {
            "header": {
                "date": f"{ep.year}-{ep.month:02d}-{ep.day:02d}",
                "season": ep.season,
                "month": ep.month_name,
                "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][ep.day_of_week],
                "total_orders": total,
                "debuffs_fired": ep.debuffs_fired,
                "picker": ep.workers[ep.picker_id].name if not ep.picker_needs_replacement else "NEEDS_REPLACEMENT",
                "is_high_volume": ep.is_high_volume,
            },
            "footer": {
                "orders_shipped": shipped,
                "orders_total": total,
                "orders_remaining": remaining,
                "reward": round(self.total_reward, 2),
                "reward_breakdown": {k: round(v, 2) for k, v in self.reward_breakdown.items() if v != 0},
                "ot_hours": round(self.ot_hours, 2),
                "restock_pct": round(restock_pct * 100, 1),
                "deliberate_complete": self.deliberate_complete,
                "filler_complete": self.filler_complete,
                "grade": grade,
            },
            "steps": self.step_log,
        }
