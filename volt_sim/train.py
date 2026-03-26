"""
Training loop for the Volt Warehouse RL simulation.

Usage:
  python volt_sim/train.py                    # fresh start
  python volt_sim/train.py --resume           # resume from latest checkpoint
  python volt_sim/train.py --resume ppo_ep900.pt  # resume from specific checkpoint
"""
import sys
import time
import os
import glob
import argparse
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volt_sim.config import TRAINING, PPO as PPO_CFG
from volt_sim.env.warehouse_env import WarehouseEnv
from volt_sim.agent.ppo import PPOAgent
from volt_sim.agent.actions import decode_actions, get_valid_action_mask, NUM_ACTION_HEADS, ACTION_HEAD_SIZE
from volt_sim.agent.state import RunningStats, TOTAL_STATE_SIZE
from volt_sim.sim_logging.episode_logger import EpisodeLogger

CHECKPOINT_DIR = "volt_sim/data/checkpoints"


def find_latest_checkpoint() -> tuple[str, int] | None:
    """Find the most recent checkpoint and extract its episode number."""
    pattern = os.path.join(CHECKPOINT_DIR, "ppo_ep*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by episode number
    def ep_num(path):
        name = os.path.basename(path)
        try:
            return int(name.replace("ppo_ep", "").replace(".pt", ""))
        except ValueError:
            return 0
    files.sort(key=ep_num)
    latest = files[-1]
    return latest, ep_num(latest)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", nargs="?", const="latest", default=None,
                        help="Resume from checkpoint. 'latest' or a filename like ppo_ep900.pt")
    args = parser.parse_args()

    env = WarehouseEnv()
    agent = PPOAgent()
    logger = EpisodeLogger()
    state_stats = RunningStats(TOTAL_STATE_SIZE)

    total_episodes = TRAINING["total_episodes"]
    log_interval = TRAINING["log_interval"]
    save_interval = TRAINING["save_interval"]
    start_episode = 1

    # Resume from checkpoint
    if args.resume:
        if args.resume == "latest":
            result = find_latest_checkpoint()
            if result:
                path, ep = result
                loaded = agent.load(path, state_stats)
                if loaded:
                    start_episode = ep + 1
                    print(f"Resumed from {os.path.basename(path)} (episode {ep})")
                else:
                    print("Checkpoint incompatible. Clearing old checkpoints.")
                    for f in glob.glob(os.path.join(CHECKPOINT_DIR, "ppo_ep*.pt")):
                        os.remove(f)
                    start_episode = 1
            else:
                print("No checkpoints found, starting fresh.")
        else:
            path = os.path.join(CHECKPOINT_DIR, args.resume)
            if os.path.exists(path):
                loaded = agent.load(path, state_stats)
                if loaded:
                    try:
                        start_episode = int(args.resume.replace("ppo_ep", "").replace(".pt", "")) + 1
                    except ValueError:
                        start_episode = 1
                    print(f"Resumed from {args.resume} (starting at episode {start_episode})")
                else:
                    print("Checkpoint incompatible, starting fresh.")
                    start_episode = 1
            else:
                print(f"Checkpoint not found: {path}")
                return

    print(f"Training episodes {start_episode} to {total_episodes}...")
    print(f"State size: {TOTAL_STATE_SIZE}")
    print(f"Action heads: {NUM_ACTION_HEADS} workers × {ACTION_HEAD_SIZE} tasks")
    print()

    start_time = time.time()

    for episode_num in range(start_episode, total_episodes + 1):
        state = env.reset()
        state_stats.update(state)
        norm_state = state_stats.normalize(state)

        episode_reward = 0.0
        done = False
        steps = 0

        while not done:
            # Get valid action mask (per-worker)
            mask = get_valid_action_mask(env)

            # Select action — one task per worker
            actions, log_prob, value = agent.select_action(norm_state, mask)

            # Decode into (worker_id, task_id) pairs
            reassignments = decode_actions(actions)

            # Step environment
            next_state, reward, done, info = env.step(reassignments)

            # Store transition
            agent.store_transition(
                norm_state, actions, log_prob, reward, value, done, mask
            )

            # Update state
            state_stats.update(next_state)
            norm_state = state_stats.normalize(next_state)
            episode_reward += reward
            steps += 1

        # Log episode
        summary = env.get_episode_summary()
        logger.log_episode(summary, episode_num)

        # Update PPO every N episodes (accumulate enough transitions)
        episodes_per_update = PPO_CFG.get("episodes_per_update", 1)
        if episode_num % episodes_per_update == 0:
            metrics = agent.update()
        else:
            metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}

        # Print progress
        if episode_num % log_interval == 0:
            footer = summary["footer"]
            header = summary["header"]
            stats = logger.get_training_stats()
            elapsed = time.time() - start_time

            print(
                f"Ep {episode_num:5d} | "
                f"{header['season']:6s} {header['month']:9s} {header['day_of_week']:9s} | "
                f"Orders: {footer['orders_shipped']}/{footer['orders_total']} | "
                f"Grade: {footer['grade']} | "
                f"Reward: {footer['reward']:8.1f} | "
                f"OT: {footer['ot_hours']:.1f}h | "
                f"AvgR100: {stats['avg_reward_last_100']:8.1f} | "
                f"WinR: {stats['win_rate_last_100']:.1%} | "
                f"PL: {metrics['policy_loss']:.4f} | "
                f"VL: {metrics['value_loss']:.4f} | "
                f"Ent: {metrics['entropy']:.4f} | "
                f"{elapsed:.0f}s"
            )

        # Save checkpoint
        if episode_num % save_interval == 0:
            os.makedirs("volt_sim/data/checkpoints", exist_ok=True)
            agent.save(f"volt_sim/data/checkpoints/ppo_ep{episode_num}.pt", state_stats)

    # Final save
    agent.save("volt_sim/data/checkpoints/ppo_final.pt", state_stats)
    print(f"\nTraining complete. {total_episodes} episodes in {time.time() - start_time:.0f}s")
    print(f"Logs saved to: volt_sim/data/episode_log.json")
    print(f"Notable episodes: volt_sim/data/notable_episodes/")


if __name__ == "__main__":
    train()
