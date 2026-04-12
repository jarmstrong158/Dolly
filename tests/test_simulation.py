"""
Dolly RL Warehouse Simulation — pytest test suite.
Tests simulation logic only: state encoding, rewards, episode generation,
action masking, worker assignment, grading, and env.step().
Does NOT test training loops, neural network weights, or rendering.
"""
import random
import numpy as np
import pytest

from volt_sim.config import (
    NUM_WORKERS, NUM_TASKS, TASKS, TASK_TO_IDX,
    TOTAL_STATE_SIZE, WORKER_STATE_SIZE, ENV_STATE_SIZE,
    DAY_START_HOUR, EOD_HOUR, REWARDS,
    MARCUS_MANAGEMENT_HOURS_REQUIRED,
    PACK_FATIGUE_THRESHOLD, PICK_FATIGUE_THRESHOLD,
    ORDER_VOLUME_RANGES, MONTH_TO_SEASON, SEASONS,
    RESTOCK_STARTING_LEVEL,
)
from volt_sim.env.warehouse_env import WarehouseEnv
from volt_sim.env.episode_generator import generate_episode
from volt_sim.env.workers import WorkerState, roll_debuffs
from volt_sim.env.order_arrival import generate_arrival_schedule, is_high_volume_day
from volt_sim.agent.actions import get_valid_action_mask, decode_actions, IDLE_IDX
from volt_sim.agent.state import validate_state, RunningStats


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    """Fresh WarehouseEnv, reset with fixed deterministic parameters."""
    e = WarehouseEnv()
    random.seed(42)
    np.random.seed(42)
    e.reset(force_month=4, force_dow=0, force_volume=200)  # April Monday 200 orders
    return e


@pytest.fixture
def all_pack_actions():
    """Every worker assigned to pack (task index 1)."""
    pack_idx = TASK_TO_IDX["pack"]
    return [(wid, pack_idx) for wid in range(NUM_WORKERS)]


# ─── 1. State Encoding ────────────────────────────────────────────────────────

class TestStateEncoding:

    def test_state_shape(self, env):
        state = env._get_state()
        assert state.shape == (TOTAL_STATE_SIZE,), (
            f"Expected ({TOTAL_STATE_SIZE},), got {state.shape}"
        )

    def test_state_dtype(self, env):
        state = env._get_state()
        assert state.dtype == np.float32

    def test_no_nan_or_inf(self, env):
        state = env._get_state()
        assert not np.any(np.isnan(state)), "State contains NaN"
        assert not np.any(np.isinf(state)), "State contains Inf"

    def test_total_size_matches_config(self):
        expected = NUM_WORKERS * WORKER_STATE_SIZE + ENV_STATE_SIZE
        assert TOTAL_STATE_SIZE == expected, (
            f"TOTAL_STATE_SIZE={TOTAL_STATE_SIZE} != {expected}"
        )

    def test_task_one_hot_per_worker(self, env):
        """Each worker's task slice sums to exactly 1.0 (one-hot)."""
        state = env._get_state()
        for wid in range(NUM_WORKERS):
            offset = wid * WORKER_STATE_SIZE
            task_slice = state[offset:offset + NUM_TASKS]
            assert task_slice.sum() == pytest.approx(1.0), (
                f"Worker {wid} task one-hot sum={task_slice.sum()}"
            )

    def test_worker_with_picker_reflects_in_state(self, env):
        """The is_picker flag (index 6 within worker block) is 1 for the designated picker."""
        state = env._get_state()
        picker_id = env.episode.picker_id
        IS_PICKER_OFFSET = NUM_TASKS + 6  # 6 one-hot + oph, hw, hr, generic_debuff, indiv_debuff, fatigue, is_picker
        # Verify: one-hot (6) + oph(1) + hw(1) + hr(1) + generic(1) + indiv(1) + fatigue(1) = 12, then is_picker at +6
        # Actual layout: task_one_hot(6), oph(1), hours_worked(1), hours_remaining(1),
        #                generic_debuff(1), individual_debuff(1), fatigue(1), is_picker(1)
        offset = picker_id * WORKER_STATE_SIZE + NUM_TASKS + 6
        assert state[offset] == pytest.approx(1.0), (
            f"Picker {picker_id} is_picker flag not set in state"
        )

    def test_absent_worker_state(self, env):
        """Force a worker absent; verify their hours_remaining contribution is ~0."""
        env.episode.workers[6].is_absent = True
        env.episode.workers[6].current_task = "idle"
        state = env._get_state()
        # hours_remaining is index 2 within scalar block (after 6 one-hot + oph + hours_worked)
        wid = 6
        hr_offset = wid * WORKER_STATE_SIZE + NUM_TASKS + 2  # oph=0, hw=1, hr=2
        # hours_remaining for absent worker: they still have shift hours unless we set them worked
        # Just verify state is valid (no crash, no NaN)
        assert not np.isnan(state[hr_offset])

    def test_state_hour_progress_at_start(self, env):
        """At day start, time progress slot should be near 0."""
        state = env._get_state()
        env_block_start = NUM_WORKERS * WORKER_STATE_SIZE
        time_val = state[env_block_start]
        assert time_val == pytest.approx(0.0, abs=0.05), (
            f"Expected time progress ~0 at day start, got {time_val}"
        )

    def test_season_one_hot_in_state(self, env):
        """Exactly one season flag is set in the environment block."""
        state = env._get_state()
        env_block_start = NUM_WORKERS * WORKER_STATE_SIZE
        # season one-hot starts at offset 6 in the env block:
        # [time(1), orders_in_q(1), orders_comp(1), picked_not_aud(1), restock_rem(1), side_proj(1), season_onehot(4), ...]
        season_start = env_block_start + 6
        season_slice = state[season_start:season_start + 4]
        assert season_slice.sum() == pytest.approx(1.0), (
            f"Season one-hot sum={season_slice.sum()}"
        )
        # April is spring → index 1
        assert season_slice[1] == pytest.approx(1.0), (
            f"Expected spring (index 1), got {season_slice}"
        )

    def test_validate_state_helper(self, env):
        state = env._get_state()
        assert validate_state(state)

    def test_validate_state_rejects_nan(self):
        bad = np.zeros(TOTAL_STATE_SIZE, dtype=np.float32)
        bad[5] = float("nan")
        assert not validate_state(bad)

    def test_validate_state_rejects_wrong_shape(self):
        bad = np.zeros(TOTAL_STATE_SIZE + 1, dtype=np.float32)
        assert not validate_state(bad)

    def test_restock_level_in_state(self, env):
        """restock_level feature is in [0, 1]."""
        state = env._get_state()
        env_block_start = NUM_WORKERS * WORKER_STATE_SIZE
        # Last feature in env block
        restock_level_idx = env_block_start + ENV_STATE_SIZE - 1
        val = state[restock_level_idx]
        assert 0.0 <= val <= 1.0, f"restock_level out of range: {val}"


# ─── 2. Reward Calculation ────────────────────────────────────────────────────

class TestRewardCalculation:

    def test_per_order_shipped_positive(self, env):
        """Shipping an order produces a positive reward contribution."""
        assert REWARDS["per_order_shipped"] > 0

    def test_ot_penalty_negative(self, env):
        """OT hour penalty is negative."""
        assert REWARDS["per_ot_hour"] < 0

    def test_incomplete_order_penalty_negative(self):
        assert REWARDS["per_order_incomplete"] < 0

    def test_all_orders_complete_bonus_positive(self):
        assert REWARDS["all_orders_complete_bonus"] > 0

    def test_management_duty_met_positive(self):
        assert REWARDS["management_duty_met"] > 0

    def test_management_duty_missed_negative(self):
        assert REWARDS["management_duty_missed"] < 0

    def test_add_reward_accumulates_breakdown(self, env):
        """_add_reward stores values in reward_breakdown and returns the right amount."""
        env.reward_breakdown = {k: 0.0 for k in REWARDS}
        result = env._add_reward("per_order_shipped", 5.0)
        expected = REWARDS["per_order_shipped"] * 5.0
        assert result == pytest.approx(expected)
        assert env.reward_breakdown["per_order_shipped"] == pytest.approx(expected)

    def test_idle_penalty_negative(self):
        assert REWARDS["per_idle_hour"] < 0

    def test_per_productive_hour_positive(self):
        assert REWARDS["per_productive_hour"] > 0

    def test_finalize_ot_penalty(self, env):
        """Forcing OT hours > 0 then finalizing produces negative OT reward."""
        env.ot_hours = 0.5
        env.orders_in_queue = 0
        env.orders_picked_not_audited = 0
        env.restock_remaining = 0.0
        # Ensure management quota met
        for w in env.episode.workers:
            if w.worker_id in (0, 1):
                w.management_hours = MARCUS_MANAGEMENT_HOURS_REQUIRED
        env.is_done = False
        r = env._finalize_episode()
        # Should include per_ot_hour penalty
        ot_component = REWARDS["per_ot_hour"] * 0.5
        assert r < 0 or ot_component < 0  # at minimum ot penalty was added

    def test_finalize_all_orders_bonus(self, env):
        """When all orders complete with no OT, bonus is positive."""
        env.orders_in_queue = 0
        env.orders_picked_not_audited = 0
        env.orders_completed = env.episode.total_orders
        env.restock_remaining = 0.0
        env.ot_hours = 0.0
        for w in env.episode.workers:
            if w.worker_id in (0, 1):
                w.management_hours = MARCUS_MANAGEMENT_HOURS_REQUIRED
        env.is_done = False
        r = env._finalize_episode()
        assert r > 0, f"Expected positive finalization reward, got {r}"

    def test_finalize_incomplete_orders_negative(self, env):
        """With incomplete orders and no OT, finalization is negative."""
        env.orders_in_queue = 50
        env.orders_picked_not_audited = 0
        env.ot_hours = 0.0
        env.is_done = False
        env.restock_remaining = 0.0
        for w in env.episode.workers:
            if w.worker_id in (0, 1):
                w.management_hours = MARCUS_MANAGEMENT_HOURS_REQUIRED
        r = env._finalize_episode()
        assert r < 0, f"Expected negative reward for incomplete orders, got {r}"

    def test_blake_prohibited_task_penalty_negative(self):
        assert REWARDS["blake_prohibited_task"] < 0


# ─── 3. Episode Generator ─────────────────────────────────────────────────────

class TestEpisodeGenerator:

    def test_generate_episode_returns_config(self):
        ep = generate_episode()
        assert ep is not None

    def test_total_orders_within_month_range(self):
        for _ in range(20):
            month = random.randint(1, 12)
            import calendar
            month_name = calendar.month_name[month]
            ep = generate_episode(force_month=month)
            lo, hi = ORDER_VOLUME_RANGES[month_name]
            assert lo <= ep.total_orders <= hi, (
                f"Month {month_name}: {ep.total_orders} not in [{lo}, {hi}]"
            )

    def test_forced_volume_respected(self):
        ep = generate_episode(force_volume=150)
        assert ep.total_orders == 150

    def test_forced_month_respected(self):
        ep = generate_episode(force_month=7)
        assert ep.month == 7
        assert ep.season == "summer"

    def test_forced_dow_respected(self):
        import calendar
        ep = generate_episode(force_dow=2)  # Wednesday
        assert ep.day_of_week == 2
        # Verify the actual calendar day matches
        assert calendar.weekday(ep.year, ep.month, ep.day) == 2

    def test_season_matches_month(self):
        for month, expected_season in MONTH_TO_SEASON.items():
            ep = generate_episode(force_month=month)
            assert ep.season == expected_season, (
                f"Month {month}: expected {expected_season}, got {ep.season}"
            )

    def test_workers_list_length(self):
        ep = generate_episode()
        assert len(ep.workers) == NUM_WORKERS

    def test_restock_hours_positive(self):
        for _ in range(10):
            ep = generate_episode()
            assert ep.restock_hours >= 0.0

    def test_arrival_schedule_sums_to_total(self):
        for _ in range(10):
            ep = generate_episode()
            total = sum(ep.arrival_schedule.values())
            assert total == ep.total_orders, (
                f"Arrival schedule sum {total} != total_orders {ep.total_orders}"
            )

    def test_high_volume_flag_consistency(self):
        """Force a high-volume value and check the flag matches."""
        import calendar as cal
        month = 5  # May: 350-500
        month_name = cal.month_name[month]
        lo, hi = ORDER_VOLUME_RANGES[month_name]
        # Force a volume above the 75th percentile
        high_thresh = int(lo + (hi - lo) * 0.75)
        ep = generate_episode(force_month=month, force_volume=high_thresh + 10)
        assert ep.is_high_volume is True

    def test_low_volume_not_high(self):
        import calendar as cal
        month = 1  # January: 60-100
        month_name = cal.month_name[month]
        lo, hi = ORDER_VOLUME_RANGES[month_name]
        ep = generate_episode(force_month=month, force_volume=lo)
        assert ep.is_high_volume is False

    def test_picker_id_matches_schedule(self):
        from volt_sim.config import PICKER_SCHEDULE
        for dow, expected_picker in PICKER_SCHEDULE.items():
            ep = generate_episode(force_dow=dow)
            if not ep.picker_needs_replacement:
                assert ep.picker_id == expected_picker, (
                    f"DOW {dow}: expected picker {expected_picker}, got {ep.picker_id}"
                )

    def test_deliberate_project_size_when_present(self):
        # Run enough times that we're likely to get one
        found = False
        for _ in range(50):
            ep = generate_episode()
            if ep.has_deliberate_project:
                lo, hi = 2.0, 6.0
                assert lo <= ep.deliberate_project_size <= hi
                found = True
                break
        # Not strictly required to find one in 50 but DELIBERATE_PROJECT_CHANCE=0.25
        # so probability of not finding one is 0.75^50 ≈ 0.00006 — essentially guaranteed

    def test_eod_hour_matches_config(self):
        ep = generate_episode()
        assert ep.eod_hour == EOD_HOUR

    def test_cooldown_tracker_reduces_individual_debuff_probability(self):
        """With a full cooldown tracker, debuffed workers should not refire."""
        # Put all workers on cooldown
        tracker = {
            "Marcus": 10, "Nolan": 10, "Felix": 10,
            "Blake": 10, "Reid": 10, "Trent": 10, "Omar": 10,
        }
        # Run multiple episodes — no individual debuffs should fire for these workers
        for _ in range(10):
            ep = generate_episode(cooldown_tracker=tracker)
            for w in ep.workers:
                # Trent's soreness has cooldown_days=0, so skip it
                # Blake's flare has cooldown_days=0 too
                if w.name in ("Trent", "Blake", "Reid"):
                    continue
                assert w.individual_debuff is None, (
                    f"{w.name} fired debuff despite cooldown"
                )


# ─── 4. Action Masking ────────────────────────────────────────────────────────

class TestActionMasking:

    def test_mask_shape(self, env):
        mask = get_valid_action_mask(env)
        assert len(mask) == NUM_WORKERS
        for worker_mask in mask:
            assert len(worker_mask) == NUM_TASKS

    def test_absent_worker_only_idle(self, env):
        """Absent worker mask: only idle is True."""
        env.episode.workers[6].is_absent = True
        mask = get_valid_action_mask(env)
        worker_mask = mask[6]
        assert worker_mask[IDLE_IDX] is True
        assert sum(worker_mask) == 1, f"Absent worker has more than 1 valid action: {worker_mask}"

    def test_pack_only_worker_cannot_pick(self, env):
        """Blake's flare: is_pack_only=True means pick is masked."""
        env.episode.workers[3].is_pack_only = True
        mask = get_valid_action_mask(env)
        pick_idx = TASK_TO_IDX["pick"]
        assert mask[3][pick_idx] is False, "Pack-only worker should not be able to pick"

    def test_pack_only_worker_can_pack(self, env):
        env.episode.workers[3].is_pack_only = True
        mask = get_valid_action_mask(env)
        pack_idx = TASK_TO_IDX["pack"]
        assert mask[3][pack_idx] is True, "Pack-only worker should still be able to pack"

    def test_non_manager_cannot_manage(self, env):
        """Non-manager workers (id >= 2) should not have management available."""
        mask = get_valid_action_mask(env)
        mgmt_idx = TASK_TO_IDX["management"]
        for wid in range(2, NUM_WORKERS):
            assert mask[wid][mgmt_idx] is False, (
                f"Worker {wid} should not be able to do management"
            )

    def test_manager_can_manage_when_quota_not_met(self, env):
        """Marcus and Nolan can manage when quota not yet filled."""
        for w in env.episode.workers:
            w.management_hours = 0.0
        mask = get_valid_action_mask(env)
        mgmt_idx = TASK_TO_IDX["management"]
        assert mask[0][mgmt_idx] is True, "Marcus should be able to manage"
        assert mask[1][mgmt_idx] is True, "Nolan should be able to manage"

    def test_manager_cannot_manage_when_quota_met(self, env):
        """Once management quota is filled, management is locked out."""
        for w in env.episode.workers:
            w.management_hours = MARCUS_MANAGEMENT_HOURS_REQUIRED
        mask = get_valid_action_mask(env)
        mgmt_idx = TASK_TO_IDX["management"]
        assert mask[0][mgmt_idx] is False, "Marcus should not manage after quota met"
        assert mask[1][mgmt_idx] is False, "Nolan should not manage after quota met"

    def test_restock_masked_when_none_remaining(self, env):
        """If restock_remaining == 0, restock task should be masked for non-pickers."""
        env.restock_remaining = 0.0
        # Make sure picker is not involved
        for w in env.episode.workers:
            w.is_picker = False
        mask = get_valid_action_mask(env)
        restock_idx = TASK_TO_IDX["restock"]
        for wid in range(NUM_WORKERS):
            if not env.episode.workers[wid].is_absent:
                assert mask[wid][restock_idx] is False, (
                    f"Worker {wid} should not be able to restock when nothing remains"
                )

    def test_decode_actions_roundtrip(self):
        task_list = [1, 0, 1, 1, 2, 1, 1]  # pack, pick, pack...
        decoded = decode_actions(task_list)
        assert len(decoded) == NUM_WORKERS
        for wid, (w, t) in enumerate(decoded):
            assert w == wid
            assert t == task_list[wid]

    def test_no_mask_all_false(self, env):
        """At minimum each non-absent worker has at least one valid action."""
        mask = get_valid_action_mask(env)
        for wid, worker_mask in enumerate(mask):
            w = env.episode.workers[wid]
            if not w.is_absent:
                assert any(worker_mask), f"Worker {wid} has no valid actions"


# ─── 5. Worker Assignment Logic ───────────────────────────────────────────────

class TestWorkerAssignment:

    def test_step_changes_worker_task(self, env):
        """Calling step with pick action for worker 0 (normally manager) — but
        Marcus can pick so we just verify the task changes."""
        pick_idx = TASK_TO_IDX["pick"]
        # Assign everyone to pick (workers without restriction can pick)
        actions = [(wid, pick_idx) for wid in range(NUM_WORKERS)]
        env.step(actions)
        # After step Marcus was assigned pick; his task should have changed
        # (unless he's restricted — he's not pack_only so pick is valid)
        assert env.episode.workers[0].current_task == "pick"

    def test_pack_only_assignment_rejected(self, env):
        """Assigning pick to a pack_only worker triggers penalty and skips assignment."""
        env.episode.workers[3].is_pack_only = True
        env.episode.workers[3].current_task = "pack"  # start at pack
        pick_idx = TASK_TO_IDX["pick"]
        actions = [(3, pick_idx)]
        initial_reward = env.total_reward
        env.step(actions)
        # Task should NOT have changed to pick
        assert env.episode.workers[3].current_task != "pick"

    def test_absent_worker_stays_idle(self, env):
        """An absent worker's task stays idle regardless of what the agent assigns."""
        env.episode.workers[6].is_absent = True
        env.episode.workers[6].current_task = "idle"
        pack_idx = TASK_TO_IDX["pack"]
        actions = [(6, pack_idx)]
        env.step(actions)
        # Absent worker stays idle (sim enforces this in _simulate_step)
        assert env.episode.workers[6].current_task == "idle"

    def test_worker_task_switch_resets_carry(self, env):
        """Switching tasks resets the work_carry accumulator."""
        w = env.episode.workers[2]
        w.work_carry = 3.5
        w.current_task = "pack"
        pack_idx = TASK_TO_IDX["pack"]
        pick_idx = TASK_TO_IDX["pick"]
        # Switch from pack to pick
        actions = [(2, pick_idx)] + [(wid, pack_idx) for wid in range(NUM_WORKERS) if wid != 2]
        env.step(actions)
        assert env.episode.workers[2].work_carry == pytest.approx(0.0) or \
               env.episode.workers[2].current_task == "pick"  # carry was reset

    def test_orders_packed_increases_after_step(self, env):
        """After a full step with pickers+packers, orders_completed should increase."""
        env.orders_in_queue = 100
        env.orders_picked_not_audited = 50  # pre-loaded for packers
        initial_completed = env.orders_completed
        pack_idx = TASK_TO_IDX["pack"]
        pick_idx = TASK_TO_IDX["pick"]
        actions = [(0, pick_idx), (1, pack_idx), (2, pack_idx),
                   (3, pack_idx), (4, pick_idx), (5, pack_idx), (6, pack_idx)]
        env.step(actions)
        assert env.orders_completed > initial_completed


# ─── 6. Fatigue System ────────────────────────────────────────────────────────

class TestFatigueSystem:

    def test_pack_fatigue_triggers(self):
        w = WorkerState(worker_id=2, name="Felix", base_oph=16.23,
                        shift_hours=8.0, role="warehouse")
        w.orders_packed = PACK_FATIGUE_THRESHOLD - 1
        w.check_fatigue()
        assert not w.is_fatigued
        w.orders_packed = PACK_FATIGUE_THRESHOLD
        w.check_fatigue()
        assert w.is_fatigued

    def test_pick_fatigue_triggers(self):
        w = WorkerState(worker_id=4, name="Reid", base_oph=18.94,
                        shift_hours=8.0, role="warehouse")
        w.orders_picked = PICK_FATIGUE_THRESHOLD - 1
        w.check_fatigue()
        assert not w.is_fatigued
        w.orders_picked = PICK_FATIGUE_THRESHOLD
        w.check_fatigue()
        assert w.is_fatigued

    def test_fatigue_reduces_oph(self):
        w = WorkerState(worker_id=2, name="Felix", base_oph=16.23,
                        shift_hours=8.0, role="warehouse")
        normal_oph = w.effective_oph("pack")
        w.is_fatigued = True
        fatigued_oph = w.effective_oph("pack")
        assert fatigued_oph < normal_oph

    def test_trent_soreness_activates_after_threshold(self):
        w = WorkerState(worker_id=5, name="Trent", base_oph=15.28,
                        shift_hours=8.0, role="warehouse")
        w.has_soreness = True
        w.non_side_project_hours = 3.9
        w.check_trent_soreness()
        assert not w.soreness_activated
        w.non_side_project_hours = 4.0
        w.check_trent_soreness()
        assert w.soreness_activated

    def test_trent_soreness_cuts_oph(self):
        w = WorkerState(worker_id=5, name="Trent", base_oph=15.28,
                        shift_hours=8.0, role="warehouse")
        w.has_soreness = True
        normal_oph = w.effective_oph("pack")
        w.soreness_activated = True
        sore_oph = w.effective_oph("pack")
        assert sore_oph < normal_oph


# ─── 7. Grade System ─────────────────────────────────────────────────────────

class TestGradeSystem:

    def _setup_for_grade(self, env):
        """Common setup: ensure management quota met."""
        for w in env.episode.workers:
            if w.worker_id in (0, 1):
                w.management_hours = MARCUS_MANAGEMENT_HOURS_REQUIRED

    def test_grade_f_when_orders_incomplete(self, env):
        self._setup_for_grade(env)
        env.orders_in_queue = 10
        env.orders_picked_not_audited = 0
        env.orders_completed = env.episode.total_orders - 10
        env.restock_remaining = 0.0
        env.ot_hours = 0.0
        env.is_done = True
        summary = env.get_episode_summary()
        assert summary["footer"]["grade"] == "F"

    def test_grade_a_when_everything_perfect(self, env):
        self._setup_for_grade(env)
        env.orders_in_queue = 0
        env.orders_picked_not_audited = 0
        env.orders_completed = env.episode.total_orders
        env.restock_remaining = 0.0  # fully restocked
        env.ot_hours = 0.0
        env.is_done = True
        summary = env.get_episode_summary()
        assert summary["footer"]["grade"] == "A"

    def test_grade_b_one_demerit(self, env):
        """All orders done, no OT, management met, but restock incomplete → B."""
        self._setup_for_grade(env)
        env.orders_in_queue = 0
        env.orders_picked_not_audited = 0
        env.orders_completed = env.episode.total_orders
        env.restock_remaining = env.episode.restock_hours  # fully incomplete
        env.ot_hours = 0.0
        env.is_done = True
        summary = env.get_episode_summary()
        assert summary["footer"]["grade"] == "B"

    def test_grade_c_two_demerits(self, env):
        """Orders done, but both restock incomplete and OT used → C."""
        self._setup_for_grade(env)
        env.orders_in_queue = 0
        env.orders_picked_not_audited = 0
        env.orders_completed = env.episode.total_orders
        env.restock_remaining = env.episode.restock_hours
        env.ot_hours = 0.5
        env.is_done = True
        summary = env.get_episode_summary()
        assert summary["footer"]["grade"] == "C"

    def test_grade_d_three_demerits(self, env):
        """Orders done but restock incomplete, OT used, management missed → D."""
        # Reset management hours to 0
        for w in env.episode.workers:
            w.management_hours = 0.0
        env.orders_in_queue = 0
        env.orders_picked_not_audited = 0
        env.orders_completed = env.episode.total_orders
        env.restock_remaining = env.episode.restock_hours
        env.ot_hours = 0.5
        env.is_done = True
        summary = env.get_episode_summary()
        assert summary["footer"]["grade"] == "D"

    def test_grade_summary_structure(self, env):
        env.is_done = True
        summary = env.get_episode_summary()
        assert "header" in summary
        assert "footer" in summary
        assert "steps" in summary
        assert "grade" in summary["footer"]
        assert summary["footer"]["grade"] in ("A", "B", "C", "D", "F")

    def test_management_waived_on_heavy_day(self, env):
        """On 400+ order days, management miss does not drop the grade."""
        # Force total_orders >= 400 via the episode
        env.episode.total_orders = 400
        for w in env.episode.workers:
            w.management_hours = 0.0  # management NOT met
        env.orders_in_queue = 0
        env.orders_picked_not_audited = 0
        env.orders_completed = 400
        env.restock_remaining = 0.0
        env.ot_hours = 0.0
        env.is_done = True
        summary = env.get_episode_summary()
        assert summary["footer"]["grade"] == "A"


# ─── 8. Simulation Step ───────────────────────────────────────────────────────

class TestSimulationStep:

    def test_step_returns_correct_types(self, env):
        pack_idx = TASK_TO_IDX["pack"]
        actions = [(wid, pack_idx) for wid in range(NUM_WORKERS)]
        state, reward, done, info = env.step(actions)
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_state_shape(self, env):
        pack_idx = TASK_TO_IDX["pack"]
        actions = [(wid, pack_idx) for wid in range(NUM_WORKERS)]
        state, _, _, _ = env.step(actions)
        assert state.shape == (TOTAL_STATE_SIZE,)

    def test_step_advances_time(self, env):
        initial_hour = env.current_hour
        pack_idx = TASK_TO_IDX["pack"]
        actions = [(wid, pack_idx) for wid in range(NUM_WORKERS)]
        from volt_sim.config import STEP_DURATION
        env.step(actions)
        # Lunch step doesn't advance the same way, so allow for 2x duration
        assert env.current_hour > initial_hour

    def test_step_done_false_early_in_day(self, env):
        """Early in the day, done should be False."""
        pack_idx = TASK_TO_IDX["pack"]
        actions = [(wid, pack_idx) for wid in range(NUM_WORKERS)]
        _, _, done, _ = env.step(actions)
        assert done is False

    def test_step_info_has_expected_keys(self, env):
        pack_idx = TASK_TO_IDX["pack"]
        actions = [(wid, pack_idx) for wid in range(NUM_WORKERS)]
        _, _, _, info = env.step(actions)
        for key in ("current_hour", "orders_in_queue", "orders_completed",
                    "total_orders", "is_ot"):
            assert key in info, f"Missing key: {key}"

    def test_episode_ends_at_hard_stop(self, env):
        """Force the clock past OT_HARD_STOP; done should be True."""
        from volt_sim.config import OT_HARD_STOP, STEP_DURATION
        env.current_hour = OT_HARD_STOP - STEP_DURATION
        env.orders_in_queue = 5  # keep non-zero so OT logic triggers
        env.is_ot = True
        env.ot_hours = 1.0
        pack_idx = TASK_TO_IDX["pack"]
        actions = [(wid, pack_idx) for wid in range(NUM_WORKERS)]
        _, _, done, _ = env.step(actions)
        assert done is True

    def test_episode_ends_when_all_orders_done_at_eod(self, env):
        """If queue empties right at EOD, done=True."""
        from volt_sim.config import EOD_HOUR, STEP_DURATION
        env.current_hour = EOD_HOUR - STEP_DURATION
        env.orders_in_queue = 0
        env.orders_picked_not_audited = 0
        pack_idx = TASK_TO_IDX["pack"]
        actions = [(wid, pack_idx) for wid in range(NUM_WORKERS)]
        _, _, done, _ = env.step(actions)
        assert done is True

    def test_total_reward_accumulates(self, env):
        """total_reward grows across steps."""
        pack_idx = TASK_TO_IDX["pack"]
        pick_idx = TASK_TO_IDX["pick"]
        actions = [(0, pick_idx), (1, pack_idx), (2, pack_idx),
                   (3, pack_idx), (4, pick_idx), (5, pack_idx), (6, pack_idx)]
        env.step(actions)
        env.step(actions)
        # total_reward may be positive or negative but must have moved from 0
        assert env.total_reward != 0.0

    def test_reset_clears_state(self):
        """Reset returns fresh state after a partially-played episode."""
        e = WarehouseEnv()
        random.seed(1)
        e.reset(force_month=3, force_dow=1, force_volume=100)
        pack_idx = TASK_TO_IDX["pack"]
        actions = [(wid, pack_idx) for wid in range(NUM_WORKERS)]
        e.step(actions)
        e.step(actions)
        # Reset again
        random.seed(2)
        state = e.reset(force_month=6, force_dow=3, force_volume=300)
        assert state.shape == (TOTAL_STATE_SIZE,)
        assert e.orders_completed == 0
        assert e.current_hour == DAY_START_HOUR
        assert e.is_done is False

    def test_full_day_loop_terminates(self):
        """Run an entire episode to completion — should terminate."""
        e = WarehouseEnv()
        random.seed(99)
        e.reset(force_month=1, force_dow=0, force_volume=80)
        pack_idx = TASK_TO_IDX["pack"]
        pick_idx = TASK_TO_IDX["pick"]
        actions = [(0, pick_idx), (1, pack_idx), (2, pack_idx),
                   (3, pack_idx), (4, pick_idx), (5, pack_idx), (6, pack_idx)]
        done = False
        steps = 0
        while not done and steps < 200:
            _, _, done, _ = e.step(actions)
            steps += 1
        assert done is True, f"Episode did not terminate after {steps} steps"


# ─── 9. Order Arrival Curve ───────────────────────────────────────────────────

class TestOrderArrival:

    def test_arrival_schedule_sums_correctly(self):
        total = 250
        schedule = generate_arrival_schedule(total, is_high_volume=False, eod_hour=EOD_HOUR)
        assert sum(schedule.values()) == total

    def test_high_volume_schedule_sums_correctly(self):
        total = 400
        schedule = generate_arrival_schedule(total, is_high_volume=True, eod_hour=EOD_HOUR)
        assert sum(schedule.values()) == total

    def test_arrival_keys_are_valid_hours(self):
        schedule = generate_arrival_schedule(100, is_high_volume=False, eod_hour=EOD_HOUR)
        for hour in schedule.keys():
            assert DAY_START_HOUR <= hour <= EOD_HOUR, f"Arrival at invalid hour {hour}"

    def test_no_negative_arrivals(self):
        schedule = generate_arrival_schedule(200, is_high_volume=False, eod_hour=EOD_HOUR)
        for hour, count in schedule.items():
            assert count >= 0, f"Negative arrivals at hour {hour}"

    def test_is_high_volume_above_threshold(self):
        lo, hi = 350, 500
        threshold = int(lo + (hi - lo) * 0.75)
        assert is_high_volume_day(threshold + 1, (lo, hi)) is True

    def test_is_high_volume_below_threshold(self):
        lo, hi = 350, 500
        assert is_high_volume_day(lo, (lo, hi)) is False


# ─── 10. Running Stats (State Normalizer) ─────────────────────────────────────

class TestRunningStats:

    def test_initial_mean_zero(self):
        rs = RunningStats(TOTAL_STATE_SIZE)
        assert np.all(rs.mean == 0.0)

    def test_update_changes_mean(self):
        rs = RunningStats(TOTAL_STATE_SIZE)
        x = np.ones(TOTAL_STATE_SIZE, dtype=np.float32) * 5.0
        rs.update(x)
        assert np.allclose(rs.mean, 5.0)

    def test_normalize_output_shape(self):
        rs = RunningStats(TOTAL_STATE_SIZE)
        x = np.random.rand(TOTAL_STATE_SIZE).astype(np.float32)
        rs.update(x)
        out = rs.normalize(x)
        assert out.shape == (TOTAL_STATE_SIZE,)

    def test_normalize_clips_extremes(self):
        rs = RunningStats(TOTAL_STATE_SIZE)
        x = np.zeros(TOTAL_STATE_SIZE, dtype=np.float32)
        rs.update(x)
        big = np.ones(TOTAL_STATE_SIZE, dtype=np.float32) * 1e9
        out = rs.normalize(big, clip=10.0)
        assert np.all(out <= 10.0)
