"""
Clean standalone PPO implementation.
Actor-critic with shared trunk, one policy head per worker (7 heads × 6 tasks).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from volt_sim.config import PPO as PPO_CFG, TOTAL_STATE_SIZE
from volt_sim.agent.actions import ACTION_HEAD_SIZE, NUM_ACTION_HEADS


class ActorCritic(nn.Module):
    def __init__(self, state_size: int = TOTAL_STATE_SIZE,
                 head_size: int = ACTION_HEAD_SIZE,
                 num_heads: int = NUM_ACTION_HEADS,
                 hidden_size: int = PPO_CFG["hidden_size"],
                 num_layers: int = PPO_CFG["num_layers"]):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        # Shared trunk
        layers = []
        in_size = state_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        self.trunk = nn.Sequential(*layers)

        # Policy heads — one per worker, each outputs task logits
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_size, head_size)
            for _ in range(num_heads)
        ])

        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        for head in self.policy_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor, action_masks: torch.Tensor = None):
        """
        state: (batch, state_size)
        action_masks: (batch, num_heads, head_size) boolean
        Returns: list of logits per head, value
        """
        features = self.trunk(state)

        logits_list = []
        for i, head in enumerate(self.policy_heads):
            logits = head(features)
            if action_masks is not None:
                mask = action_masks[:, i, :]
                logits = logits.masked_fill(~mask, float("-inf"))
            logits_list.append(logits)

        value = self.value_head(features).squeeze(-1)
        return logits_list, value

    def get_action(self, state: np.ndarray, action_mask: list[list[bool]] = None):
        """
        Sample actions for a single state.
        action_mask: list of NUM_HEADS lists, each of length HEAD_SIZE
        Returns: actions (list of int), log_prob (tensor), value (tensor)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)

        if action_mask is not None:
            mask_t = torch.BoolTensor(action_mask).unsqueeze(0)  # (1, num_heads, head_size)
        else:
            mask_t = None

        with torch.no_grad():
            logits_list, value = self.forward(state_t, mask_t)

        actions = []
        log_probs = []
        for logits in logits_list:
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))

        log_prob = torch.stack(log_probs).sum()
        return actions, log_prob, value.squeeze(0)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor,
                         action_masks: torch.Tensor = None):
        """
        Evaluate log probs and entropy for batched states/actions.
        states: (batch, state_size)
        actions: (batch, num_heads)
        action_masks: (batch, num_heads, head_size)
        """
        logits_list, values = self.forward(states, action_masks)

        total_log_prob = torch.zeros(states.shape[0], device=states.device)
        total_entropy = torch.zeros(states.shape[0], device=states.device)

        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            total_log_prob += dist.log_prob(actions[:, i])
            total_entropy += dist.entropy()

        return total_log_prob, values, total_entropy


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []

    def add(self, state, action, log_prob, reward, value, done, action_mask=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def compute_returns(self, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def get_batches(self, batch_size: int, advantages: np.ndarray,
                    returns: np.ndarray):
        """Yield minibatches for PPO updates."""
        n = len(self.states)
        indices = np.random.permutation(n)

        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array([lp.item() for lp in self.log_probs])

        has_masks = self.action_masks[0] is not None
        if has_masks:
            masks = np.array(self.action_masks)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch = {
                "states": torch.FloatTensor(states[idx]),
                "actions": torch.LongTensor(actions[idx]),
                "old_log_probs": torch.FloatTensor(old_log_probs[idx]),
                "advantages": torch.FloatTensor(advantages[idx]),
                "returns": torch.FloatTensor(returns[idx]),
            }

            if has_masks:
                batch["action_masks"] = torch.BoolTensor(masks[idx])
            else:
                batch["action_masks"] = None

            yield batch

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()


class PPOAgent:
    def __init__(self):
        self.model = ActorCritic()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=PPO_CFG["lr"]
        )
        self.buffer = RolloutBuffer()

        self.clip_epsilon = PPO_CFG["clip_epsilon"]
        self.entropy_coeff = PPO_CFG["entropy_coeff"]
        self.value_loss_coeff = PPO_CFG["value_loss_coeff"]
        self.max_grad_norm = PPO_CFG["max_grad_norm"]
        self.epochs = PPO_CFG["epochs_per_update"]
        self.batch_size = PPO_CFG["batch_size"]
        self.gamma = PPO_CFG["gamma"]
        self.gae_lambda = PPO_CFG["gae_lambda"]

    def select_action(self, state: np.ndarray, action_mask: list[list[bool]] = None):
        actions, log_prob, value = self.model.get_action(state, action_mask)
        return actions, log_prob, value.item()

    def store_transition(self, state, action, log_prob, reward, value, done,
                         action_mask=None):
        if action_mask is not None:
            mask_array = np.array(action_mask)  # (num_heads, head_size)
        else:
            mask_array = None
        self.buffer.add(state, action, log_prob, reward, value, done, mask_array)

    def update(self) -> dict:
        advantages, returns = self.buffer.compute_returns(self.gamma, self.gae_lambda)

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "n_updates": 0}

        for _ in range(self.epochs):
            for batch in self.buffer.get_batches(self.batch_size, advantages, returns):
                log_probs, values, entropy = self.model.evaluate_actions(
                    batch["states"], batch["actions"], batch["action_masks"]
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch["returns"])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                        self.value_loss_coeff * value_loss +
                        self.entropy_coeff * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += -entropy_loss.item()
                metrics["n_updates"] += 1

        # Average metrics
        if metrics["n_updates"] > 0:
            for k in ("policy_loss", "value_loss", "entropy"):
                metrics[k] /= metrics["n_updates"]

        self.buffer.clear()
        return metrics

    def save(self, path: str, state_stats=None):
        data = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if state_stats is not None:
            data["state_stats"] = {
                "n": state_stats.n,
                "mean": state_stats.mean,
                "var": state_stats.var,
                "_m2": state_stats._m2,
            }
        torch.save(data, path)

    def load(self, path: str, state_stats=None):
        checkpoint = torch.load(path, weights_only=False)
        try:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except (RuntimeError, ValueError) as e:
            print(f"Checkpoint incompatible (dimensions changed), starting fresh: {e}")
            return False
        if state_stats is not None and "state_stats" in checkpoint:
            ss = checkpoint["state_stats"]
            state_stats.n = ss["n"]
            state_stats.mean = ss["mean"]
            state_stats.var = ss["var"]
            state_stats._m2 = ss.get("_m2", ss["var"] * max(1, ss["n"] - 1))
        return True
