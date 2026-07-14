# Assignment 4 Specification

## Learning Objectives

By completing this assignment, you should be able to:

1. translate policy-gradient, Bellman-error, and actor-critic equations into PyTorch;
2. distinguish on-policy trajectory learning from off-policy replay-based learning;
3. explain why DQN needs both replay and a delayed target network;
4. diagnose unstable training from reward and loss curves;
5. compare algorithms under a controlled, reproducible protocol.

## Required Implementation

### Part A: Shared Networks

Complete `PolicyNet`, `QNet`, and `CriticNet` in `models.py`. Each network must accept a single state or a batch of states and return correctly shaped outputs. `PolicyNet` must produce a valid action distribution or logits used consistently by the policy agents.

### Part B: REINFORCE

Complete `agents/pg_agent.py`:

- initialize the policy and optimizer;
- sample actions from the policy distribution;
- retain the information needed for the score-function gradient;
- compute discounted returns after an episode;
- update the policy and clear the trajectory buffer;
- save and load model state.

### Part C: DQN

Complete `agents/dqn_agent.py`:

- initialize online and target Q-networks and an optimizer;
- implement epsilon-greedy action selection;
- add complete transitions to replay;
- sample minibatches only when enough transitions exist;
- compute a detached Bellman target with terminal masking;
- optimize the online network and synchronize the target network;
- save and load model state.

### Part D: Actor-Critic

Complete `agents/ac_agent.py`:

- initialize actor and critic networks and optimizers;
- sample actions from the actor;
- compute the one-step TD target and TD error;
- update the critic from value error;
- update the actor using a detached advantage signal;
- save and load both networks.

The starter signatures are intentionally minimal. You may add private fields and helper methods and may pass additional transition data through the existing `store(...)` call sites. Keep the public methods expected by `Trainer` intact.

## Experiments

Run all three algorithms on `CartPole-v1` with a shared episode budget. Use at least three random seeds for the final comparison. Include:

- learning curves with a moving average;
- final evaluation return for each seed;
- one ablation for DQN (replay or target-network synchronization);
- one learning-rate comparison for either REINFORCE or Actor-Critic;
- a concise interpretation connecting observed behavior to the update equations.

Do not claim that one method is universally superior from this single environment.

## Submission

Submit the completed source files, an executed copy of `notebook.ipynb`, generated figures, and a short report. Do not submit virtual environments, caches, or large intermediate checkpoints.
