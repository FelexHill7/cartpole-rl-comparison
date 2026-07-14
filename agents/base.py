from abc import ABC, abstractmethod
from typing import Tuple, List
import gymnasium as gym
import torch

class Agent(ABC):
    agent_key = None
    def __init__(self, config, device, env_name='CartPole-v1'):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.gamma = config.get("gamma", 0.98)
        self.lr = config.get("lr", 5e-4)

    @abstractmethod
    def select_action(self, state) -> Tuple[int, dict]:
        pass

    @abstractmethod
    def store(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def train(self, num_train_episodes) -> Tuple:
        pass

    @abstractmethod
    def evaluate(self, out_path, num_episodes) -> List[float]:
        pass

    @abstractmethod
    def save(self, path) -> None:
        pass

    @abstractmethod
    def load(self, path) -> None:
        pass