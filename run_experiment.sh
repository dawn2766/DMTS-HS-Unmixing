#!/bin/bash

# DMTS-Net实验运行脚本

echo "=========================================="
echo "DMTS-Net Experiment Runner"
echo "=========================================="

# 激活虚拟环境（如果使用）
# source dmts_env/bin/activate

# 创建结果目录
mkdir -p results
mkdir -p data

# 运行完整流程
echo ""
echo "Running full pipeline (training + evaluation)..."
python main.py \
    --mode both \
    --n_endmembers 4 \
    --K 3 \
    --batch_size 32 \
    --lr 1e-4 \
    --ee_epochs 100 \
    --abundance_epochs 100 \
    --save_dir ./results \
    --cuda

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Results saved to ./results/"
echo "=========================================="
