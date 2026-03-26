"""
Log schema definitions for episode logging.
"""


def episode_log_entry(episode_summary: dict, episode_num: int) -> dict:
    """Build a complete log entry from an episode summary."""
    return {
        "episode": episode_num,
        "header": episode_summary["header"],
        "footer": episode_summary["footer"],
        "steps": episode_summary["steps"],
    }


def notable_episode_entry(episode_summary: dict, episode_num: int,
                          reason: str) -> dict:
    """Build a notable episode log entry."""
    entry = episode_log_entry(episode_summary, episode_num)
    entry["notable_reason"] = reason
    return entry
