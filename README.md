# CartPole RL Comparison

This repository contains a complete implementation and comparison of three reinforcement-learning agents on CartPole-v1:

- REINFORCE (policy gradient)
- DQN (value-based, replay-buffer-based)
- Actor-Critic (advantage-based policy/value learning)

The project includes:
- completed agent implementations in [agents](agents)
- shared neural network definitions in [models.py](models.py)
- a training/evaluation pipeline in [trainer.py](trainer.py)
- an interactive walkthrough notebook in [notebook.ipynb](notebook.ipynb)

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
- The notebook has been updated to work on Windows and Unix-based systems.
