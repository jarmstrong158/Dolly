# VOLT WAREHOUSE RL SIMULATION — FULL BUILD

Build a warehouse operations reinforcement learning simulation from scratch. This is a standalone project — do not inherit anything from Balatron. Create a new repo structure with clean separation between the simulation environment, the PPO agent, the logger, and the dashboard.

## PROJECT STRUCTURE

```
volt_sim/
  env/
    warehouse_env.py
    workers.py
    episode_generator.py
    order_arrival.py
  agent/
    ppo.py
    state.py
    actions.py
  logging/
    episode_logger.py
    log_schema.py
  dashboard/
    dashboard.html
  data/
    workers.json
  train.py
  config.py
```

## WORKER ROSTER

```python
workers = [
    {"name": "Marcus", "oph": 17.0, "shift_hours": 9.75, "role": "manager"},
    {"name": "Nolan", "oph": 15.35, "shift_hours": 8.0, "role": "assistant_manager"},
    {"name": "Felix", "oph": 16.23, "shift_hours": 8.0, "role": "warehouse"},
    {"name": "Blake", "oph": 18.30, "shift_hours": 8.0, "role": "warehouse"},
    {"name": "Reid", "oph": 18.94, "shift_hours": 8.0, "role": "warehouse"},
    {"name": "Trent", "oph": 15.28, "shift_hours": 8.0, "role": "warehouse"},
    {"name": "Omar", "oph": 14.88, "shift_hours": 8.0, "role": "warehouse"},
]
```

## PICKER SCHEDULE (fixed input, not agent decision)

- Monday: Reid
- Tuesday: Blake
- Wednesday: Felix
- Thursday: Omar
- Friday: Trent

If Blake has a flare debuff on Tuesday, the agent selects a replacement picker from all available workers including Marcus and Nolan. Marcus and Nolan's order metric penalties still apply if selected.

## SEASONAL ORDER VOLUME

```python
order_volume_ranges = {
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
```

## SEASON DERIVATION

- Winter: December, January, February
- Spring: March, April, May
- Summer: June, July, August
- Fall: September, October, November

## ORDER ARRIVAL CURVE

Orders are not all present at day start. Distribute total daily volume across time buckets with noise:

- 9:00 AM (day start): 40-50% of total — leftover from prior day
- 9:00 AM — 2:00 PM: steady flow, ~25-30% of total
- 2:00 PM — 4:15 PM: slow period, ~10-15% of total
- 4:15 PM — EOD: late surge, ~15-20% of total

High volume days (top 25% of seasonal range): flatten the curve, distribute evenly throughout the day.

## RESTOCK FORMULA

```python
restock_hours = 2.5 + (order_volume * 0.01) + random.uniform(-0.5, 0.5)
```

Restock demand is positively correlated with order volume. Higher order days generate more restock need.

## SIDE PROJECTS

Two types generated at episode start:

- **Deliberate** (25% chance): real task with a size in worker-hours, higher reward weight
- **Filler** (always available): infinite buffer — deep cleaning, organizing, painting — very small reward, never depletes

## DECISION POINTS

1. **Day start (9:00 AM):** Full initial worker assignment based on complete episode state
2. **Every 30 minutes:** Agent may reassign up to 2 workers
3. **Lunch (1:00 PM):** Hard 30-minute stop for all workers simultaneously, full mid-day reassessment
4. **EOD:** Episode ends, OT triggers automatically if orders remain incomplete

## ACTION SPACE

Each 30-minute interval the agent selects up to 2 workers to reassign. Action = (worker_id, new_task). Valid tasks: pick, pack, restock, side_project, idle. The agent cannot assign Blake to pick, restock, or side_project when his flare debuff is active — this is a hard constraint enforced by the environment, not the agent.

## STATE VECTOR (~65 variables)

**Per worker (7 workers × 8 values = 56):**

- Current assignment (one-hot: 5 tasks)
- Effective OPH (normalized)
- Hours worked today
- Hours remaining in shift
- Active generic debuff flag
- Active individual debuff flag
- Fatigue flag (hit pack/pick threshold)
- Is today's picker (boolean)

**Environment (9 values):**

- Current hour (normalized 0-1)
- Orders remaining in queue
- Orders completed so far today
- Orders picked but not yet audited
- Restock tasks remaining
- Side project progress (0-1)
- Season (one-hot: 4)
- Is high volume day (boolean)
- Marcus's management hours completed today

## DEBUFF SYSTEM

Debuffs are rolled in two independent categories at episode start. A worker receives exactly one result per category. Individual debuffs roll separately and stack on top.

### SLEEP CATEGORY (mutually exclusive, roll one per worker)

| Result | OPH Multiplier | Probability |
|---|---|---|
| Well rested | 1.05 | 10% |
| Normal | 1.00 | 75% |
| Bad sleep | 0.95 | 15% |

### HEALTH CATEGORY (mutually exclusive, roll one per worker)

| Result | OPH Multiplier | Probability |
|---|---|---|
| Locked in | 1.20 | 5% |
| Normal | 1.00 | 69% |
| Bad headspace | See preferences | 8% |
| Sick mild | 0.85 | 5% |
| Very sick | 0.60 | 1% |
| Injured | 0.50 | 2% |
| (buffer) | 1.00 | 10% |

### BAD HEADSPACE BY WORKER

| Worker | Effect |
|---|---|
| Marcus | -5% all tasks |
| Nolan | +10% side projects, -10% all others |
| Felix | +10% packing, -10% all others |
| Reid | Immune — treat as Normal |
| Omar | +10% packing, -10% all others |
| Blake | +10% packing, -10% all others |
| Trent | +10% packing, -10% all others |

### INDIVIDUAL DEBUFFS (roll independently, stack on top of category results)

| Worker | Debuff | Model | Effect |
|---|---|---|---|
| Marcus | Family needs | 25% daily after 5-day cooldown | Loses 1-2 hrs labor |
| Nolan | Family needs | 25% daily after 5-day cooldown | Loses 1-2 hrs labor |
| Felix | Stomach issues | 30% daily after 4-day cooldown | -15% OPH |
| Blake | EOE/muscle flare | Season-weighted, no cooldown | Pack only for the day |
| Reid | No call no show | 2.5% flat daily, no cooldown | Loses entire day |
| Trent | Soreness | 16.7% flat daily | -50% OPH after 4hrs non-side-project work |
| Omar | Family needs | 15% daily after 7-day cooldown | Loses 1-2 hrs labor |

### ANDREW SEASONAL FLARE PROBABILITIES

- Winter: 5%
- Spring: 20%
- Summer: 25%
- Fall: 12%

## EFFECTIVE OPH FORMULA

```python
effective_oph = base_oph * sleep_modifier * health_modifier * individual_modifier * fatigue_modifier
```

## GENERAL FATIGUE (all workers)

After completing 110 pack orders OR 230 pick orders in a single day, OPH drops by 15% for the remainder of the shift. Trent's soreness mechanic is separate and stacks on top if active.

## JONATHAN — MANAGER CONSTRAINTS

- 4 hours management duty required daily (fixed, non-negotiable)
- Remaining hours flex to restock, side projects, or order support
- Small negative reinforcement per order completed personally

## REWARD SIGNALS

### Orders

- +1.0 per order shipped
- +50.0 flat bonus if all orders completed by EOD
- -10.0 per order incomplete at EOD
- -0.5 per OT hour authorized
- -25.0 flat + -10.0 per order if orders remain incomplete despite OT
- -0.3 per order Marcus completes personally
- -0.15 per order Nolan completes personally

### Restock

- +0.3 per restock task completed
- +10.0 flat if all restock done by EOD
- -0.5 per restock task bleeding to next day
- -3.0 flat if restock backlog causes a pick interruption

### Side projects

- +0.1 per unit of filler work completed
- +5.0 flat bonus for filler project completion
- +0.1 per unit of deliberate project work completed
- +8.0 flat bonus for deliberate project completion
- -2.0 flat if agent assigns side project during active order crunch

### Worker management

- +0.2 per productive worker-hour
- -0.1 per idle worker-hour
- +20.0 flat if Marcus's 4hr management duty is met
- -15.0 flat if management duty is not met
- -5.0 flat if Blake is assigned a prohibited task (hard constraint violation)

## LOGGING

Write `episode_log.json` after each episode. Rolling window of 100 episodes. Save notable episodes separately: best reward, worst reward, most debuffs fired, first perfect day (all orders complete, no OT).

### Per time step

- Timestamp (simulated time)
- Each worker: assignment, effective OPH, active debuffs
- Orders remaining, completed, picked-not-audited
- Restock tasks remaining
- Side project progress
- Running reward score

### Per episode header

- Simulated date, season, month, day of week
- Total order volume for the day
- All debuffs fired at episode start
- Today's picker

### Per episode footer

- Orders shipped vs total
- Final reward score with full signal breakdown
- OT hours if any
- Restock completion percentage
- Side project completion
- Grade: A (≥90% max reward), B (75-89%), C (60-74%), D (45-59%), F (<45%)

## DASHBOARD (dashboard.html)

Single self-contained HTML file. Uses D3.js and Chart.js from cdnjs.cloudflare.com. Reads `episode_log.json` directly — no server required, open in browser after training run.

### Visualizations

1. **Worker timeline** — horizontal bar per worker, color coded by task across the simulated day
2. **Order queue depth** — line graph showing order arrivals vs completions over the day
3. **Running reward curve** for the current episode
4. **Episode summary panel** — grade, orders shipped, OT hours, restock %, debuffs fired
5. **Training progress panel** — last 100 episodes: win rate trend, average reward, OT frequency

## PPO IMPLEMENTATION

Clean standalone PPO. Do not reference or inherit from Balatron. Standard actor-critic with shared trunk, separate policy head and value head. Observation space is the ~65 variable state vector. Action space is discrete per-worker reassignment at each 30-minute interval.

## CONSTRAINTS

- All tunable parameters must live in `config.py` — reward values, debuff probabilities, OPH data, seasonal ranges, cooldown lengths, fatigue thresholds. Nothing hardcoded.
- Debuff conflict rules must be enforced at roll time — a worker cannot be both well rested and bad sleep, or both locked in and sick.
- Blake's prohibited task constraint must be enforced by the environment, not relied on from the agent.
- Dashboard must work by simply opening `dashboard.html` in a browser after a training run with no additional setup.
