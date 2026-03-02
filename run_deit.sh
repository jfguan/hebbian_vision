#!/bin/bash
# Wait for current training to finish, then run DeiT recipe (d256, 8 layers, 4.82M params)
while pgrep -f "train.py.*cifar100_d256" > /dev/null 2>&1; do
    echo "$(date): training still running, checking again in 10m..."
    sleep 600
done
echo "$(date): training finished, starting DeiT run (d256, 8 layers, 4.82M params)"
uv run python train.py --d-model 256 --n-layers 8 --tag deit_d256_n8 --epochs 200
