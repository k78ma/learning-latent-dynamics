#!/bin/bash

. .venv/bin/activate
CONFIG=exp_baseline
for seed in 4; do
  echo using seed $seed
  python train.py --config $CONFIG --seed $seed
done