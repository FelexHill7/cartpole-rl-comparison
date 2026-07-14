import os
import copy
import torch
import torch.nn.functional as F
from models import QNet
from replay_buffer import ReplayBuffer
from agents.base import Agent
from moviepy import ImageSequenceClip


class DQNAgent(Agent):
    """Deep Q-Network with replay buffer and a delayed target network."""

    agent_key = "dqn"

    def __init__(self, config, env_name, device=None):
        super().__init__(config=config, env_name=env_name, device=device)

        self.epsilon = config.get("epsilon", 0.1)
        self.buffer_size = config.get("buffer_size", 10000)
        self.batch_size = config.get("batch_size", 32)
        self.sync_interval = config.get("sync_interval", 20)
        self.train_interval = config.get("train_interval", 1)

        self.qnet = QNet(self.OBS_DIM, self.ACT_DIM, hidden=config.get("hidden", 128)).to(self.device)
        self.target_qnet = copy.deepcopy(self.qnet).to(self.device)
        self.target_qnet.requires_grad_(False)
        self.target_qnet.eval()
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

        self.replay = ReplayBuffer(self.buffer_size, self.batch_size, device=self.device)
        self.global_steps = 0

    OBS_DIM = 4
    ACT_DIM = 2

    def select_action(self, state):
        if torch.rand(1).item() < self.epsilon:
            action = int(self.env.action_space.sample())
            return action, {}

        with torch.no_grad():
            state = state.unsqueeze(0) if state.dim() == 1 else state
            q_values = self.qnet(state.to(self.device))
            action = int(torch.argmax(q_values, dim=1).item())
        return action, {}

    def store(self, state, action, reward, next_state, terminated, **kwargs):
        self.replay.add(
            state.detach().cpu(),
            int(action),
            float(reward),
            next_state.detach().cpu(),
            bool(terminated),
        )

    def update(self):
        self.global_steps += 1
        if self.global_steps % self.train_interval != 0:
            return {"loss": 0.0}
        if len(self.replay) < self.batch_size:
            return {"loss": 0.0}

        states, actions, rewards, next_states, terminates = self.replay.minibatch()
        q_values = self.qnet(states).gather(1, actions.unsqueeze(1).long()).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_qnet(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - terminates.float()) * next_q_values

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def synchronize(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def train(self, state, episode):
        terminated = False
        truncated = False
        total_reward = 0
        loss = {"loss": 0.0}
        while not terminated and not truncated:
            action, info = self.select_action(state)
            next_s, reward, terminated, truncated, _ = self.env.step(action)
            next_state = torch.tensor(next_s, device=self.device).float()
            done = terminated or truncated
            self.store(state, action, reward, next_state, done)
            loss = self.update()
            state = next_state
            total_reward += reward

        if episode % self.sync_interval == 0:
            try:
                self.synchronize()
            except Exception:
                pass
        return {"reward": total_reward, "loss": loss["loss"] if loss else None}

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.qnet.state_dict(), os.path.join(path, "qnet.pt"))

    def load(self, path):
        self.qnet.load_state_dict(torch.load(os.path.join(path, "qnet.pt"), map_location=self.device))
        self.qnet.eval()
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()

    def evaluate(self, out_path, num_episodes=1):
        rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device).float()
            terminated = False
            truncated = False
            total = 0
            frames = []
            while not terminated and not truncated:
                frames.append(self.env.render())
                with torch.no_grad():
                    qs = self.qnet(state.to(self.device))
                    action = int(torch.argmax(qs, dim=-1).item())
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                state = torch.tensor(next_state, device=self.device).float()
                total += reward
            rewards.append(total)
            clip = ImageSequenceClip(sequence=frames, fps=24)
            clip.write_videofile(os.path.join(out_path, f"./{self.agent_key}_evaluate_{episode}.mp4"), codec="libx264")
        return rewards