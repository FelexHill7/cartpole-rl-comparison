import os
import torch
import numpy as np

from agents.base import Agent
from agents.pg_agent import PGAgent
from agents.dqn_agent import DQNAgent
from agents.ac_agent import ACAgent
from utils import plot_rewards, plot_compare, plot_loss

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

class Trainer:
    AGENTS = {_cls.agent_key: _cls for _cls in [PGAgent, DQNAgent, ACAgent]}

    def __init__(self, device=None, hyperparams_map=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: {}'.format(self.device))
        self.hyperparams_map = hyperparams_map
        for key in self.AGENTS.keys():
            assert key in self.hyperparams_map, f"Missing hyperparams for agent {key}"

    def train(self, agent_key, env_name, out_dir, config=None, num_episodes=100):
        assert agent_key in self.AGENTS, f"Unknown agent {agent_key}"
        if not config:
            config = self.hyperparams_map.get(agent_key, {})
        out_dir = os.path.join(out_dir, agent_key)
        os.makedirs(out_dir, exist_ok=True)
        AgentCls = self.AGENTS[agent_key]
        agent: Agent = AgentCls(config=config, env_name=env_name, device=self.device)

        reward_history = []
        loss_history = []

        for episode in range(num_episodes):
            state, _ = agent.env.reset()
            state = torch.tensor(state, device=self.device).float()

            metric = agent.train(state, episode)

            total_reward = metric['reward']
            loss_info = metric['loss']
            if isinstance(loss_info, dict):
                if "policy_loss" in loss_info and "critic_loss" in loss_info:
                    loss_value = loss_info["policy_loss"]
                else:
                    loss_value = None
            else:
                loss_value = loss_info

            reward_history.append(total_reward)
            loss_history.append(loss_value)
            print(f"[{agent_key}] Episode {episode} reward={total_reward}")

        # save logs and model
        plot_rewards(reward_history, f"{agent_key} rewards", os.path.join(out_dir, f"rewards.png"))
        np.save(os.path.join(out_dir, f"rewards.npy"), np.array(reward_history))
        if len(loss_history) > 0:
            plot_loss(loss_history, f"{agent_key} loss", os.path.join(out_dir, f"loss.png"))
            np.save(os.path.join(out_dir, f"loss.npy"), np.array(loss_history))

        # save model(s)
        agent.save(out_dir)

        return {"rewards": reward_history, "losses": loss_history}

    def evaluate(self, agent_key, env_name, model_path, num_episodes=5):
        assert agent_key in self.AGENTS, f"Unknown agent {agent_key}"
        AgentCls = self.AGENTS[agent_key]
        agent: Agent = AgentCls(config={}, env_name=env_name, device=self.device)
        model_path = os.path.join(model_path, agent_key)
        agent.load(model_path)
        return agent.evaluate(out_path=model_path, num_episodes=num_episodes)

    def compare(self, agent_keys, env_name, out_dir, num_train_episodes=100, num_eval_episodes=5):
        results = {}
        for key in agent_keys:
            print(f"Training {key}...")
            res = self.train(agent_key=key, env_name=env_name, out_dir=out_dir, config=self.hyperparams_map.get(key, {}), num_episodes=num_train_episodes)
            eval_rewards = self.evaluate(agent_key=key, env_name=env_name, model_path=out_dir, num_episodes=num_eval_episodes)
            results[key] = {
                'train_reward': res["rewards"],
                'eval_reward': eval_rewards
            }
        for key in agent_keys:
            print('[{}] evaluation rewards: {} mean={}'.format(key, results[key]['eval_reward'], np.mean(results[key]['eval_reward'])))

        plot_compare({k: v['train_reward'] for k, v in results.items()}, "Algorithm comparison", os.path.join(out_dir, "compare_rewards.png"))
        return results
