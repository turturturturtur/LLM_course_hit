#!/bin/bash
# 激活 conda 环境并启动 benchmark
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate course

cd /home/vepfs/LLM_course
python run_benchmark.py
