"""REINFORCE (Policy-Gradient) agent — fully implemented baseline.

This module is the **reference implementation** students can study before
completing DQN (agents/dqn_agent.py) and Actor-Critic (agents/ac_agent.py).
Do not modify this file unless you have a good reason.
"""

import os
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from models import PolicyNet
from agents.base import Agent
from moviepy import ImageSequenceClip


class PGAgent(Agent):
    """REINFORCE with full episode return and score-function gradient.

    The update rule is::

        G_t = sum_{k=t}^{T} gamma^{k-t} * r_{k+1}          (discounted return)
        loss = -sum_t G_t * log pi_theta(a_t | s_t)          (policy gradient)

    References
    ----------
    Williams (1992), "Simple Statistical Gradient-Following Algorithms …"
    """

    agent_key = "pg"

    # CartPole-v1 fixed dimensions
    OBS_DIM = 4
    ACT_DIM = 2

    def __init__(self, config, env_name="CartPole-v1", device=None):
        super().__init__(config=config, env_name=env_name, device=device)

        hidden = config.get("hidden", 128)
        self.policy = PolicyNet(self.OBS_DIM, self.ACT_DIM, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # Trajectory buffer cleared after every episode
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

    # ------------------------------------------------------------------
    # Core interface methods
    # ------------------------------------------------------------------

    def select_action(self, state: torch.Tensor):
        """Sample an action from the current policy; retain the log-prob."""
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), {"log_prob": log_prob}

    def store(self, reward: float, log_prob: torch.Tensor, **kwargs) -> None:
        """Append one (reward, log_prob) pair to the trajectory buffer."""
        self._rewards.append(reward)
        self._log_probs.append(log_prob)

    def update(self) -> dict:
        """Run a single REINFORCE update and clear the trajectory buffer."""
        T = len(self._rewards)
        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # Normalise returns for training stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(self._log_probs)
        loss = -(log_probs * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._log_probs.clear()
        self._rewards.clear()

        return {"loss": loss.item()}

    def train(self, state: torch.Tensor, episode: int = None) -> dict:
        """Collect one episode, run the policy update, and return metrics."""
        terminated = False
        truncated = False
        total_reward = 0.0

        while not terminated and not truncated:
            action, info = self.select_action(state)
            next_s, reward, terminated, truncated, _ = self.env.step(action)
            next_state = torch.tensor(next_s, device=self.device, dtype=torch.float32)
            self.store(reward=reward, log_prob=info["log_prob"])
            state = next_state
            total_reward += reward

        metrics = self.update()
        return {"reward": total_reward, "loss": metrics["loss"]}

    # ------------------------------------------------------------------
    # Evaluation & persistence
    # ------------------------------------------------------------------

    def evaluate(self, out_path: str, num_episodes: int = 1):
        """Run greedy evaluation and save an MP4 per episode."""
        rewards = []
        os.makedirs(out_path, exist_ok=True)
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            terminated = truncated = False
            total = 0.0
            frames = []
            while not terminated and not truncated:
                frames.append(self.env.render())
                with torch.no_grad():
                    logits = self.policy(state)
                    action = int(torch.argmax(logits).item())   # greedy
                next_s, reward, terminated, truncated, _ = self.env.step(action)
                state = torch.tensor(next_s, device=self.device, dtype=torch.float32)
                total += reward
            rewards.append(total)
            clip = ImageSequenceClip(sequence=frames, fps=24)
            clip.write_videofile(
                os.path.join(out_path, f"pg_evaluate_{ep}.mp4"), codec="libx264",
                logger=None,
            )
        return rewards

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, "policy.pt"))

    def load(self, path: str) -> None:
        self.policy.load_state_dict(
            torch.load(os.path.join(path, "policy.pt"), map_location=self.device)
        )
        self.policy.eval()
