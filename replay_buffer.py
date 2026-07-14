import random
from collections import deque

import torch


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device=None):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def add(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def minibatch(self):
        data = random.sample(self.buffer, self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        terminates = []

        for d in data:
            states.append(d[0])
            actions.append(d[1])
            rewards.append(d[2])
            next_states.append(d[3])
            terminates.append(d[4])

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        terminates = torch.tensor(terminates).int().to(self.device)
        return states, actions, rewards, next_states, terminates

    def __len__(self):
        return len(self.buffer)
