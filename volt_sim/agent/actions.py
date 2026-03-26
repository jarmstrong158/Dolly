"""
Action space encoding/decoding for the warehouse environment.

Each step the agent assigns a task to EVERY worker.
7 workers × 6 tasks = 7 independent action heads, each outputting 0-5.
"""
from volt_sim.config import (
    NUM_WORKERS, NUM_TASKS, TASKS, TASK_TO_IDX,
    MARCUS_MANAGEMENT_HOURS_REQUIRED as MGMT_REQUIRED,
)

# Each head has NUM_TASKS choices (pick/pack/restock/side_project/management/idle)
ACTION_HEAD_SIZE = NUM_TASKS  # 6
NUM_ACTION_HEADS = NUM_WORKERS  # 7

IDLE_IDX = TASK_TO_IDX["idle"]
MGMT_IDX = TASK_TO_IDX["management"]

# Only Marcus (0) and Nolan (1) can do management
MANAGEMENT_ELIGIBLE = {0, 1}


def decode_actions(action_list: list[int]) -> list[tuple[int, int]]:
    """
    Decode list of per-worker task indices into (worker_id, task_id) pairs.
    action_list: [task_idx_for_worker_0, task_idx_for_worker_1, ...]
    """
    assignments = []
    for worker_id, task_idx in enumerate(action_list):
        assignments.append((worker_id, task_idx))
    return assignments


def get_valid_action_mask(env) -> list[list[bool]]:
    """
    Returns a per-worker mask: list of NUM_WORKERS lists, each of length NUM_TASKS.
    mask[worker_id][task_id] = True if that worker can do that task.
    """
    mask = []
    for w_id in range(NUM_WORKERS):
        worker = env.episode.workers[w_id]

        if worker.is_absent:
            # Absent workers can only idle
            worker_mask = [False] * NUM_TASKS
            worker_mask[IDLE_IDX] = True
        elif worker.hours_remaining <= 0 and not env.is_ot:
            # Shift over, no OT — can only idle
            worker_mask = [False] * NUM_TASKS
            worker_mask[IDLE_IDX] = True
        elif worker.is_picker:
            # Designated picker: always can pick, other tasks available when queue empty
            worker_mask = [False] * NUM_TASKS
            worker_mask[TASK_TO_IDX["pick"]] = True
            if env.orders_in_queue == 0:
                worker_mask[TASK_TO_IDX["pack"]] = True
                if env.restock_remaining > 0:
                    worker_mask[TASK_TO_IDX["restock"]] = True
                worker_mask[TASK_TO_IDX["side_project"]] = True
        else:
            worker_mask = []
            for t_id in range(NUM_TASKS):
                task = TASKS[t_id]
                if task == "idle":
                    # Idle only for absent workers (handled above)
                    worker_mask.append(False)
                elif task == "management":
                    # Only Marcus/Nolan, shared quota — check total team mgmt hours
                    total_mgmt = sum(env.episode.workers[i].management_hours
                                     for i in MANAGEMENT_ELIGIBLE)
                    can_manage = (w_id in MANAGEMENT_ELIGIBLE and
                                  total_mgmt < MGMT_REQUIRED)
                    worker_mask.append(can_manage)
                elif worker.is_pack_only and task not in ("pack",):
                    worker_mask.append(False)
                elif task == "restock" and env.restock_remaining <= 0:
                    worker_mask.append(False)
                else:
                    worker_mask.append(True)

        mask.append(worker_mask)
    return mask
