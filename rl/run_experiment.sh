#!/bin/bash
# RLHF 实验一键运行脚本 (8卡分布式)
# 使用说明: bash run_experiment.sh [模型路径]

set -e

MODEL_NAME="${1:-../.cache/Qwen3-0.6B}"
OUTPUT_DIR="../output/rlhf"
DATASET_CACHE="../.cache/datasets"

NPROC_PER_NODE=8
MASTER_PORT=29500

echo "========================================"
echo "RLHF Experiment Pipeline (Distributed)"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE"
echo "========================================"

# ---------- 第一步：训练奖励模型 ----------
echo ""
echo ">>> Step 1: Training Reward Model..."
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=1 \
    --master_port=$MASTER_PORT \
    main.py \
    --mode train_rm \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_cache_dir "$DATASET_CACHE" \
    --rm_epochs 1 \
    --rm_lr 1e-5 \
    --rm_batch_size 16 \
    --rm_max_length 512 \
    --device cuda

echo ">>> Reward Model training completed."

# ---------- 第二步：PPO 策略对齐 ----------
echo ""
echo ">>> Step 2: PPO Policy Alignment..."
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=1 \
    --master_port=$MASTER_PORT \
    main.py \
    --mode train_ppo \
    --model_name "$MODEL_NAME" \
    --reward_model_path "$OUTPUT_DIR/rm_final" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_cache_dir "$DATASET_CACHE" \
    --ppo_steps 100 \
    --ppo_batch_size 8 \
    --ppo_mini_batch_size 4 \
    --ppo_target_kl 0.1 \
    --ppo_kl_penalty kl \
    --ppo_max_new_tokens 40 \
    --device cuda

echo ">>> PPO training completed."

# ---------- 完成 ----------
echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results located at: $OUTPUT_DIR"
echo "  - RM history:     $OUTPUT_DIR/rm/rm_history.json"
echo "  - PPO logs:       $OUTPUT_DIR/ppo/ppo_logs.json"
echo "  - PPO curves:     $OUTPUT_DIR/ppo/reward_kl_curve.png"
echo "  - Final model:    $OUTPUT_DIR/ppo/final_ppo_model"
echo "========================================"
