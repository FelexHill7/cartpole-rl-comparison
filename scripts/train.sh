#!/bin/bash
export PYTHONPATH=$(dirname "$0")/../:$PYTHONPATH

python run.py --cmd train --algo pg --episodes 1000
python run.py --cmd eval --algo pg

# python run.py --cmd train --algo dqn --episodes 1000 --batch_size 64 --buffer_size 5000
# python run.py --cmd eval --algo dqn

# python run.py --cmd train --algo ac --episodes 1000 --critic_lr 1e-3 --actor_lr 1e-3
# python run.py --cmd eval --algo ac