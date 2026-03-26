"""
Volt Warehouse RL Simulation — Central Configuration
All tunable parameters live here. Nothing hardcoded elsewhere.
"""

# ─── Worker Roster ───────────────────────────────────────────────────────────
WORKERS = [
    {"id": 0, "name": "Marcus", "oph": 17.0,  "shift_hours": 9.75, "role": "manager"},
    {"id": 1, "name": "Nolan",     "oph": 15.35, "shift_hours": 8.0,  "role": "assistant_manager"},
    {"id": 2, "name": "Felix",    "oph": 16.23, "shift_hours": 8.0,  "role": "warehouse"},
    {"id": 3, "name": "Blake",   "oph": 18.30, "shift_hours": 8.0,  "role": "warehouse"},
    {"id": 4, "name": "Reid",      "oph": 18.94, "shift_hours": 8.0,  "role": "warehouse"},
    {"id": 5, "name": "Trent",      "oph": 15.28, "shift_hours": 8.0,  "role": "warehouse"},
    {"id": 6, "name": "Omar",    "oph": 14.88, "shift_hours": 8.0,  "role": "warehouse"},
]

NUM_WORKERS = len(WORKERS)

# ─── Tasks ───────────────────────────────────────────────────────────────────
TASKS = ["pick", "pack", "restock", "side_project", "management", "idle"]
TASK_TO_IDX = {t: i for i, t in enumerate(TASKS)}
NUM_TASKS = len(TASKS)

# ─── Morning Pick Round ──────────────────────────────────────────────────────
# Everyone picks 1-2 carts at day start before assignments kick in.
# 12-14 carts total across the team, 1-6 orders per cart.
# Fixed mechanic, not an agent decision.
MORNING_PICK_CARTS_MIN = 1    # min carts per worker
MORNING_PICK_CARTS_MAX = 2    # max carts per worker
MORNING_PICK_PER_CART_MIN = 1  # min orders per cart
MORNING_PICK_PER_CART_MAX = 6  # max orders per cart

# ─── Hustle Mode ─────────────────────────────────────────────────────────────
# On high-volume days the team knows it's a push day — fewer breaks, more focus.
# Flat OPH multiplier applied to all workers on high-volume days.
HUSTLE_MODE_BONUS = 1.12  # +12% OPH across the board

# ─── Task OPH Multipliers ────────────────────────────────────────────────────
# Base OPH represents packing speed. Picking is inherently faster.
# Main picker (designated for the day) gets 2.5x, supplemental pickers get 2.25x.
PICK_MULTIPLIER_MAIN = 2.5       # today's designated picker
PICK_MULTIPLIER_SUPPLEMENT = 2.25  # everyone else picking
TASK_OPH_MULTIPLIER = {
    "pack": 1.0,          # base OPH = packing rate
    "restock": 1.0,       # restock measured in hours, not orders
    "side_project": 1.0,  # side projects measured in hours
    "management": 0.0,    # management is time-based, not output-based
    "idle": 0.0,
}
# Note: "pick" is NOT in TASK_OPH_MULTIPLIER — it's resolved dynamically
# based on whether the worker is the designated picker or supplementing.

# ─── Picker Schedule (day_of_week 0=Monday) ─────────────────────────────────
PICKER_SCHEDULE = {
    0: 4,  # Monday: Reid
    1: 3,  # Tuesday: Blake
    2: 2,  # Wednesday: Felix
    3: 6,  # Thursday: Omar
    4: 5,  # Friday: Trent
}

# ─── Seasonal Order Volume ──────────────────────────────────────────────────
ORDER_VOLUME_RANGES = {
    "January":   (60, 100),
    "February":  (60, 100),
    "March":     (150, 250),
    "April":     (300, 450),
    "May":       (350, 500),
    "June":      (350, 500),
    "July":      (280, 400),
    "August":    (280, 400),
    "September": (250, 350),
    "October":   (200, 300),
    "November":  (150, 220),
    "December":  (60, 100),
}

MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring",  4: "spring", 5: "spring",
    6: "summer",  7: "summer", 8: "summer",
    9: "fall",   10: "fall",  11: "fall",
}

SEASONS = ["winter", "spring", "summer", "fall"]

# ─── Order Arrival Curve ────────────────────────────────────────────────────
# (bucket_label, fraction_low, fraction_high)
ORDER_ARRIVAL_BUCKETS = [
    ("day_start",    0.40, 0.50),  # 9:00 AM instant
    ("morning",      0.25, 0.30),  # 9:00 AM - 2:00 PM
    ("afternoon",    0.10, 0.15),  # 2:00 PM - 4:15 PM
    ("late_surge",   0.15, 0.20),  # 4:15 PM - 5:00 PM (order cutoff)
]

HIGH_VOLUME_PERCENTILE = 0.75  # top 25% of range = high volume day

# ─── Shift Timing ───────────────────────────────────────────────────────────
DAY_START_HOUR = 9.0          # 9:00 AM (Marcus arrives at 7:45 but pre-sim hours handled via MARCUS_PRE_SIM_MANAGEMENT)
LUNCH_HOUR = 13.0             # 1:00 PM
LUNCH_DURATION = 0.5          # 30 minutes
EOD_HOUR = 17.5               # 5:30 PM — everyone leaves, including Marcus
ORDER_CUTOFF_HOUR = 17.0      # 5:00 PM — orders after this don't need completing today
STEP_DURATION = 0.25          # 15-minute intervals

# Morning arrival ends, afternoon starts, late surge starts
MORNING_END_HOUR = 14.0       # 2:00 PM
AFTERNOON_END_HOUR = 16.25    # 4:15 PM

# ─── Restock ─────────────────────────────────────────────────────────────────
RESTOCK_BASE_HOURS = 2.5
RESTOCK_VOLUME_COEFF = 0.008    # max 7.0h at 500 orders
RESTOCK_NOISE_RANGE = (-0.5, 0.5)

# Restock level system — restock is a resource that depletes as orders are picked.
# Starts at 100%. Each picked order drains it. When it hits 0%, picking speed
# drops drastically (shelves empty). Restocking refills the level.
RESTOCK_STARTING_LEVEL = 1.0            # 100%
RESTOCK_DRAIN_PER_ORDER = None          # calculated per-episode: 1.0 / total_orders * drain_factor
RESTOCK_DRAIN_FACTOR = 2.0              # level drains to 0 after ~50% of orders without restocking
RESTOCK_PICK_PENALTY_THRESHOLD = 0.2    # below 20%, picking OPH drops
RESTOCK_PICK_PENALTY_MULTIPLIER = 0.25  # at 0% restock, picking is 25% speed (shelves empty)
RESTOCK_REFILL_PER_HOUR = None          # calculated per-episode based on restock_hours

# ─── Side Projects ──────────────────────────────────────────────────────────
DELIBERATE_PROJECT_CHANCE = 0.25
DELIBERATE_PROJECT_SIZE_RANGE = (2.0, 6.0)  # worker-hours
FILLER_PROJECT_SIZE = float("inf")           # never depletes
FILLER_COMPLETION_THRESHOLD = 4.0            # hours of filler work to count as "complete"

# ─── Fatigue Thresholds ─────────────────────────────────────────────────────
PACK_FATIGUE_THRESHOLD = 110    # orders packed before fatigue kicks in
PICK_FATIGUE_THRESHOLD = 230    # orders picked before fatigue kicks in
FATIGUE_OPH_PENALTY = 0.85     # 15% drop

# Trent's soreness
TRENT_SORENESS_HOUR_THRESHOLD = 4.0   # hours of non-side-project work
TRENT_SORENESS_OPH_PENALTY = 0.50     # 50% OPH drop

# ─── Marcus Manager Constraints ───────────────────────────────────────────
MARCUS_MANAGEMENT_HOURS_REQUIRED = 4.0
MARCUS_PRE_SIM_MANAGEMENT = 1.25         # 7:45 to 9:00 = management before sim starts

# ─── Debuff: Sleep Category ─────────────────────────────────────────────────
SLEEP_DEBUFFS = [
    {"name": "well_rested", "multiplier": 1.05, "probability": 0.10},
    {"name": "normal",      "multiplier": 1.00, "probability": 0.75},
    {"name": "bad_sleep",   "multiplier": 0.95, "probability": 0.15},
]

# ─── Debuff: Health Category ────────────────────────────────────────────────
HEALTH_DEBUFFS = [
    {"name": "locked_in",     "multiplier": 1.20, "probability": 0.05},
    {"name": "normal",        "multiplier": 1.00, "probability": 0.69},
    {"name": "bad_headspace", "multiplier": None,  "probability": 0.08},  # per-worker
    {"name": "sick_mild",     "multiplier": 0.85, "probability": 0.05},
    {"name": "very_sick",     "multiplier": 0.60, "probability": 0.01},
    {"name": "injured",       "multiplier": 0.50, "probability": 0.02},
    {"name": "buffer",        "multiplier": 1.00, "probability": 0.10},
]

# ─── Bad Headspace Effects (per worker name) ────────────────────────────────
# Format: {task: multiplier} — tasks not listed get the default multiplier
BAD_HEADSPACE_EFFECTS = {
    "Marcus": {"default": 0.95},                                          # -5% all
    "Nolan":     {"side_project": 1.10, "default": 0.90},                    # +10% side, -10% rest
    "Felix":    {"pack": 1.10, "default": 0.90},                            # +10% pack, -10% rest
    "Reid":      {"default": 1.00},                                          # immune
    "Omar":    {"pack": 1.10, "default": 0.90},
    "Blake":   {"pack": 1.10, "default": 0.90},
    "Trent":      {"pack": 1.10, "default": 0.90},
}

# ─── Individual Debuffs ─────────────────────────────────────────────────────
INDIVIDUAL_DEBUFFS = {
    "Marcus": {
        "name": "family_needs",
        "probability": 0.25,
        "cooldown_days": 5,
        "effect": "lose_hours",
        "hours_lost_range": (1.0, 2.0),
    },
    "Nolan": {
        "name": "family_needs",
        "probability": 0.25,
        "cooldown_days": 5,
        "effect": "lose_hours",
        "hours_lost_range": (1.0, 2.0),
    },
    "Felix": {
        "name": "stomach_issues",
        "probability": 0.30,
        "cooldown_days": 4,
        "effect": "oph_penalty",
        "oph_multiplier": 0.85,
    },
    "Blake": {
        "name": "eoe_muscle_flare",
        "probability": None,  # season-weighted, see below
        "cooldown_days": 0,   # no cooldown
        "effect": "pack_only",
    },
    "Reid": {
        "name": "no_call_no_show",
        "probability": 0.025,
        "cooldown_days": 0,
        "effect": "absent",
    },
    "Trent": {
        "name": "soreness",
        "probability": 1.0 / 6.0,  # 16.7%
        "cooldown_days": 0,
        "effect": "soreness",
    },
    "Omar": {
        "name": "family_needs",
        "probability": 0.15,
        "cooldown_days": 7,
        "effect": "lose_hours",
        "hours_lost_range": (1.0, 2.0),
    },
}

# Blake flare probabilities by season
BLAKE_FLARE_PROBABILITIES = {
    "winter": 0.05,
    "spring": 0.20,
    "summer": 0.25,
    "fall":   0.12,
}

# ─── Reward Signals ─────────────────────────────────────────────────────────
REWARDS = {
    # Orders
    "per_order_shipped":              1.0,
    "all_orders_complete_bonus":     50.0,
    "per_order_incomplete":         -10.0,
    "per_ot_hour":                   -0.5,
    "ot_incomplete_flat":           -25.0,
    "ot_per_order_incomplete":      -10.0,
    "marcus_per_order":            -0.3,
    "nolan_per_order":                -0.15,

    # Restock — Marcus/Nolan should be preferred restockers
    "per_restock_completed":          0.3,
    "all_restock_bonus":             10.0,
    "per_restock_bleed":             -0.5,
    "restock_pick_interruption":     -3.0,
    "warehouse_worker_restock":      -0.2,  # penalty for using non-manager on restock

    # Side projects
    "per_filler_unit":                0.1,
    "filler_completion_bonus":        5.0,
    "per_deliberate_unit":            0.1,
    "deliberate_completion_bonus":    8.0,
    "side_project_during_crunch":    -2.0,

    # Worker management — idle penalty must be strong enough to discourage
    # leaving Marcus/Nolan idle after management quota is met
    "per_productive_hour":            0.3,
    "per_idle_hour":                 -0.5,
    "packers_starved":               -1.0,   # per packer with nothing to pack while queue has orders
    "picked_backlog":                -0.5,   # per 10 orders sitting picked but not packed
    "management_duty_met":           20.0,
    "management_duty_missed":       -30.0,
    "blake_prohibited_task":        -5.0,
}

# ─── Grading ───────────────────────────────────────────────────────────────
# Outcome-based demerit system (implemented in warehouse_env.py):
# F = any orders incomplete. Always.
# A = all orders + restock done + management duty met + no OT
# Each breach (restock, management, OT) drops one letter grade.

# ─── PPO Hyperparameters ────────────────────────────────────────────────────
PPO = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coeff": 0.01,
    "value_loss_coeff": 0.5,
    "max_grad_norm": 0.5,
    "epochs_per_update": 4,
    "batch_size": 64,
    "episodes_per_update": 3,    # collect 3 episodes (~100 transitions) before PPO update
    "hidden_size": 128,
    "num_layers": 2,
}

# ─── Training ───────────────────────────────────────────────────────────────
TRAINING = {
    "total_episodes": 100000,
    "log_interval": 10,
    "save_interval": 100,
    "rolling_window": 100,
}

# ─── State Vector Dimensions ────────────────────────────────────────────────
# Per worker: 6 one-hot task + 10 scalars = 16
# Scalars: oph, hours_worked, hours_remaining, generic_debuff, individual_debuff,
#          fatigue, is_picker, is_pack_only, soreness_progress, management_hours
WORKER_STATE_SIZE = NUM_TASKS + 10  # 16
ENV_STATE_SIZE = 15     # 1 hour + 1 orders_remaining + 1 orders_completed +
                        # 1 picked_not_audited + 1 restock_remaining +
                        # 1 side_project_progress + 4 season_onehot +
                        # 1 is_high_volume + 1 total_mgmt_hours + 1 picker_needs_replacement +
                        # 1 restock_level
TOTAL_STATE_SIZE = NUM_WORKERS * WORKER_STATE_SIZE + ENV_STATE_SIZE  # 7*16+15 = 127

# ─── OT ─────────────────────────────────────────────────────────────────────
# Everyone can stay until 6:30 PM (1 hour past 5:30 EOD).
# 7 workers × 1 hour = up to 7 worker-hours of OT available.
OT_WALL_CLOCK_MAX = 1.0  # 1 hour of wall clock OT (5:30 → 6:30)
OT_HARD_STOP = 18.5      # 6:30 PM — absolute latest anyone works
