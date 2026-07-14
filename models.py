import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """Two-layer MLP that maps a state observation to action logits.

    Used by REINFORCE (pg_agent) and the Actor in Actor-Critic (ac_agent).

    Args:
        obs_dim  (int): dimensionality of the state vector (4 for CartPole-v1).
        act_dim  (int): number of discrete actions (2 for CartPole-v1).
        hidden   (int): width of the single hidden layer.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, act_dim)."""
        return self.net(x)


class QNet(nn.Module):
    """Action-value network used by DQN."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape (batch, act_dim)."""
        return self.net(x)


class CriticNet(nn.Module):
    """Scalar state-value network used by actor-critic."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return value estimates of shape (batch, 1)."""
        return self.net(x)
