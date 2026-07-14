# CartPole RL Comparison

This repository provides a complete, runnable implementation and comparison of three foundational reinforcement-learning algorithms on the CartPole-v1 environment. It is designed as both a practical coding project and an educational walkthrough for understanding how policy-gradient, value-based, and actor-critic methods differ in practice.

## Project overview

CartPole is a classic control benchmark in which an agent must learn to balance a pole on a moving cart. Although the task is simple compared with modern robotics problems, it is still rich enough to illustrate core reinforcement-learning ideas such as exploration, bootstrapping, reward-to-go estimation, policy updates, and stability in optimization.

This project demonstrates how to:

- implement and train a REINFORCE policy-gradient agent
- build a Deep Q-Network with experience replay and a target network
- train a one-step Actor-Critic method using TD error and advantage estimation
- compare learning behavior across algorithms using a shared evaluation protocol

## What is included

- a guided notebook in [notebook.ipynb](notebook.ipynb) that walks through the assignment and experiments
- shared neural network definitions in [models.py](models.py)
- modular agent implementations in [agents/pg_agent.py](agents/pg_agent.py), [agents/dqn_agent.py](agents/dqn_agent.py), and [agents/ac_agent.py](agents/ac_agent.py)
- reusable training and evaluation logic in [trainer.py](trainer.py)
- command-line entry points in [run.py](run.py) for training and comparison runs
- generated outputs such as learning curves, policy-confidence plots, and rendered policy snapshots under the [output](output) directory

## Algorithms covered

1. REINFORCE
   - Monte Carlo policy-gradient learning
   - uses discounted returns to update the policy
   - provides a strong baseline for understanding variance and policy optimization

2. DQN
   - off-policy value-based learning
   - uses epsilon-greedy exploration, replay memory, and a target network
   - highlights the role of stability in Bellman updates

3. Actor-Critic
   - combines policy and value learning
   - updates the actor using a critic-based advantage signal
   - reduces variance compared with pure Monte Carlo returns

## Quick start

```bash
python -m pip install -r requirements.txt
python run.py --cmd train --algo pg --episodes 10
python run.py --cmd compare --episodes 50 --out_dir output/final_comparison
```

## Project structure

- [models.py](models.py) — policy, Q-value, and critic networks
- [agents/pg_agent.py](agents/pg_agent.py) — REINFORCE implementation
- [agents/dqn_agent.py](agents/dqn_agent.py) — DQN implementation with replay and target network
- [agents/ac_agent.py](agents/ac_agent.py) — one-step actor-critic implementation
- [trainer.py](trainer.py) — shared training, evaluation, and comparison logic
- [notebook.ipynb](notebook.ipynb) — notebook walkthrough and experiment workspace

## Notes

- Generated outputs such as plots, videos, and checkpoints are written under the [output](output) directory.
- The notebook and training workflow have been designed to work on both Windows and Unix-based systems.
