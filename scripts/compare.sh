#!/bin/bash
export PYTHONPATH=$(dirname "$0")/../:$PYTHONPATH
python run.py --cmd compare --episodes 500
