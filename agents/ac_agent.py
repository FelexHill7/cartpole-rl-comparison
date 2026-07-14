import os
import torch
from torch.distributions.categorical import Categorical
from models import PolicyNet, CriticNet
from agents.base import Agent
from moviepy import ImageSequenceClip


class ACAgent(Agent):
    """One-step actor-critic with a learned baseline value function."""

    agent_key = "ac"

    def __init__(self, config, env_name, device=None):
        super().__init__(config=config, env_name=env_name, device=device)

        self.critic_lr = config.get("critic_lr", self.lr)
        self.actor = PolicyNet(self.OBS_DIM, self.ACT_DIM, hidden=config.get("hidden", 128)).to(self.device)
        self.critic = CriticNet(self.OBS_DIM, hidden=config.get("hidden", 128)).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self._states = []
        self._next_states = []
        self._rewards = []
        self._actions = []
        self._dones = []

    OBS_DIM = 4
    ACT_DIM = 2

    def select_action(self, state):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), {"log_prob": log_prob}

    def store(self, reward, info=None, state=None, next_state=None, terminated=False, **kwargs):
        self._states.append(state.to(self.device))
        self._next_states.append(next_state.to(self.device))
        self._rewards.append(float(reward))
        self._actions.append(int(kwargs.get("action", info.get("action", 0))))
        self._dones.append(bool(terminated))

    def update(self):
        if not self._states:
            return {"policy_loss": 0.0, "critic_loss": 0.0}

        states = torch.stack(self._states).to(self.device)
        next_states = torch.stack(self._next_states).to(self.device)
        rewards = torch.tensor(self._rewards, dtype=torch.float32, device=self.device)
        actions = torch.tensor(self._actions, dtype=torch.long, device=self.device)
        dones = torch.tensor(self._dones, dtype=torch.float32, device=self.device)

        values = self.critic(states).squeeze(-1)
        next_values = self.critic(next_states).squeeze(-1)
        targets = rewards + self.gamma * (1.0 - dones) * next_values
        td_errors = targets.detach() - values

        critic_loss = td_errors.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logits = self.actor(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        advantages = td_errors.detach()
        policy_loss = -(selected_log_probs * advantages).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self._states.clear()
        self._next_states.clear()
        self._rewards.clear()
        self._actions.clear()
        self._dones.clear()
        return {"policy_loss": policy_loss.item(), "critic_loss": critic_loss.item()}

    def train(self, state, episode=None):
        terminated = False
        truncated = False
        total_reward = 0
        while not terminated and not truncated:
            action, info = self.select_action(state)
            next_s, reward, terminated, truncated, _ = self.env.step(action)
            next_state = torch.tensor(next_s, device=self.device).float()
            self.store(reward=reward, info=info, state=state, next_state=next_state, terminated=terminated or truncated, action=action)
            state = next_state
            total_reward += reward

        metrics = self.update()
        return {
            "reward": total_reward,
            "loss": {
                "policy_loss": metrics.get("policy_loss"),
                "critic_loss": metrics.get("critic_loss"),
            },
        }

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pt"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pt"), map_location=self.device))
        self.actor.eval()
        self.critic.eval()

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
                action, _ = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                state = torch.tensor(next_state, device=self.device).float()
                total += reward
            rewards.append(total)
            clip = ImageSequenceClip(sequence=frames, fps=24)
            clip.write_videofile(os.path.join(out_path, f"./{self.agent_key}_evaluate_{episode}.mp4"), codec="libx264")
        return rewards
