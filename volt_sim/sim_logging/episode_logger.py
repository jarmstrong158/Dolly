"""
Episode logger — writes episode_log.json with rolling window.
Saves notable episodes separately.
"""
import json
import os
import shutil
from pathlib import Path

from volt_sim.config import TRAINING
from volt_sim.sim_logging.log_schema import episode_log_entry, notable_episode_entry


class EpisodeLogger:
    def __init__(self, log_dir: str = "volt_sim/data"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.log_dir / "episode_log.json"
        self.notable_dir = self.log_dir / "notable_episodes"
        self.notable_dir.mkdir(exist_ok=True)

        self.rolling_window = TRAINING["rolling_window"]
        self.episodes: list[dict] = []

        # Notable tracking
        self.best_reward = float("-inf")
        self.worst_reward = float("inf")
        self.most_debuffs = 0
        self.first_perfect: dict | None = None

        # Training stats
        self.all_rewards: list[float] = []
        self.all_grades: list[str] = []
        self.ot_count: int = 0
        self.total_episodes: int = 0

        # Load existing data if present
        self._load_existing()

    def _load_existing(self):
        """Load previous training data so we can continue iterating."""
        if not self.log_path.exists():
            return
        try:
            with open(self.log_path) as f:
                data = json.load(f)
            self.episodes = data.get("episodes", [])[-self.rolling_window:]
            # Rebuild stats from loaded episodes
            for ep in self.episodes:
                reward = ep["footer"]["reward"]
                grade = ep["footer"]["grade"]
                ot = ep["footer"]["ot_hours"]
                self.all_rewards.append(reward)
                self.all_grades.append(grade)
                self.total_episodes += 1
                if ot > 0:
                    self.ot_count += 1
                if reward > self.best_reward:
                    self.best_reward = reward
                if reward < self.worst_reward:
                    self.worst_reward = reward
                debuffs = len(ep["header"].get("debuffs_fired", []))
                if debuffs > self.most_debuffs:
                    self.most_debuffs = debuffs
                if (self.first_perfect is None and
                        ep["footer"].get("orders_remaining", 1) == 0 and
                        ot == 0):
                    self.first_perfect = ep
            print(f"Loaded {len(self.episodes)} existing episodes from log.")
        except (json.JSONDecodeError, KeyError):
            print("Existing log corrupt or incompatible, starting fresh.")
            self.episodes = []

    def log_episode(self, episode_summary: dict, episode_num: int):
        entry = episode_log_entry(episode_summary, episode_num)
        self.episodes.append(entry)

        # Maintain rolling window
        if len(self.episodes) > self.rolling_window:
            self.episodes = self.episodes[-self.rolling_window:]

        # Update stats
        reward = episode_summary["footer"]["reward"]
        grade = episode_summary["footer"]["grade"]
        ot_hours = episode_summary["footer"]["ot_hours"]

        self.all_rewards.append(reward)
        self.all_grades.append(grade)
        self.total_episodes += 1
        if ot_hours > 0:
            self.ot_count += 1

        # Check for notable episodes
        num_debuffs = len(episode_summary["header"]["debuffs_fired"])

        if reward > self.best_reward:
            self.best_reward = reward
            self._save_notable(episode_summary, episode_num, "best_reward")

        if reward < self.worst_reward:
            self.worst_reward = reward
            self._save_notable(episode_summary, episode_num, "worst_reward")

        if num_debuffs > self.most_debuffs:
            self.most_debuffs = num_debuffs
            self._save_notable(episode_summary, episode_num, "most_debuffs")

        # Perfect day: all orders complete, no OT
        if (self.first_perfect is None and
                episode_summary["footer"]["orders_remaining"] == 0 and
                ot_hours == 0):
            self.first_perfect = entry
            self._save_notable(episode_summary, episode_num, "first_perfect_day")

        # Write rolling log
        self._write_log()

    def _save_notable(self, episode_summary: dict, episode_num: int, reason: str):
        entry = notable_episode_entry(episode_summary, episode_num, reason)
        path = self.notable_dir / f"{reason}.json"
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)

    def _write_log(self):
        output = {
            "episodes": self.episodes,
            "training_stats": self.get_training_stats(),
        }
        # Serialize once, then write the complete string in one shot.
        # This minimizes the window where the file is incomplete.
        data = json.dumps(output, indent=2)
        with open(self.log_path, "w") as f:
            f.write(data)

    def get_training_stats(self) -> dict:
        recent = self.all_rewards[-self.rolling_window:]
        recent_grades = self.all_grades[-self.rolling_window:]

        win_count = sum(1 for g in recent_grades if g in ("A", "B"))
        avg_reward = sum(recent) / max(1, len(recent))
        recent_ot = sum(
            1 for ep in self.episodes
            if ep["footer"]["ot_hours"] > 0
        )

        return {
            "total_episodes": self.total_episodes,
            "avg_reward_last_100": round(avg_reward, 2),
            "win_rate_last_100": round(win_count / max(1, len(recent_grades)), 3),
            "ot_frequency_last_100": round(recent_ot / max(1, len(self.episodes)), 3),
            "best_reward_ever": round(self.best_reward, 2),
            "worst_reward_ever": round(self.worst_reward, 2),
            "grade_distribution": {
                g: recent_grades.count(g) for g in ("A", "B", "C", "D", "F")
            },
        }
