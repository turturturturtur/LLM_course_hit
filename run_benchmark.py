#!/usr/bin/env python
"""
自动遍历不同并发数（GPU 数）与 ZeRO Stage，运行训练并收集显存与吞吐量指标。
"""
import json
import os
import subprocess
import sys
import glob
import pandas as pd

# ==================== 配置区 ====================
# 并发数列表（即使用的 GPU 数量）
NUM_GPUS_LIST = [1, 2, 4, 8]
# ZeRO Stage 列表
ZERO_STAGES = [0, 1, 2, 3]
# 每个 GPU 上的 micro batch size（与 DataLoader 的 batch_size 保持一致）
BATCH_SIZE = 32
# 为了快速 benchmark，只跑 1 个 epoch
EPOCHS = 1
# 结果保存目录
RESULT_DIR = "benchmark_results"

DS_CONFIG_TEMPLATE = {
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    },
    "fp16": {"enabled": True},
    "bf16": {"enabled": False},
    "zero_optimization": {"stage": 0},
}
# ================================================


def generate_ds_config(zero_stage):
    """生成对应 ZeRO stage 的 DeepSpeed 配置文件。"""
    config = DS_CONFIG_TEMPLATE.copy()
    config["zero_optimization"] = {"stage": zero_stage}
    config_path = f"/tmp/ds_config_stage{zero_stage}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def clear_old_results(num_gpus, zero_stage):
    """清理该组合之前留下的结果文件。"""
    pattern = os.path.join(
        RESULT_DIR, f"result_gpus{num_gpus}_stage{zero_stage}_rank*.json"
    )
    for f in glob.glob(pattern):
        os.remove(f)


def run_experiment(num_gpus, zero_stage, port=29500):
    """执行一次 benchmark 实验。"""
    config_path = generate_ds_config(zero_stage)
    clear_old_results(num_gpus, zero_stage)

    cmd = [
        "deepspeed",
        f"--num_gpus={num_gpus}",
        f"--master_port={port}",
        "main.py",
        "--deepspeed",
        f"--deepspeed_config={config_path}",
        f"--batch_size={BATCH_SIZE}",
        f"--epochs={EPOCHS}",
        "--max_steps=50",
    ]

    print(f"\n{'='*60}")
    print(f"正在运行实验: GPUs={num_gpus}, ZeRO Stage={zero_stage}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["BENCHMARK_GPUS"] = str(num_gpus)
    env["BENCHMARK_STAGE"] = str(zero_stage)
    env["MASTER_PORT"] = str(port)

    process = subprocess.run(cmd, env=env)
    return process.returncode


def collect_results(num_gpus, zero_stage):
    """收集某次实验的所有 rank 结果并汇总。"""
    pattern = os.path.join(
        RESULT_DIR, f"result_gpus{num_gpus}_stage{zero_stage}_rank*.json"
    )
    files = glob.glob(pattern)
    if not files:
        return None

    peak_memories = []
    throughputs = []
    total_steps = 0
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        peak_memories.append(data["peak_memory_gb"])
        throughputs.append(data["avg_throughput_tok_per_sec"])
        total_steps += data.get("num_steps", 0)

    return {
        "num_gpus": num_gpus,
        "zero_stage": zero_stage,
        "avg_peak_memory_gb": round(sum(peak_memories) / len(peak_memories), 2),
        "max_peak_memory_gb": round(max(peak_memories), 2),
        "min_peak_memory_gb": round(min(peak_memories), 2),
        "per_gpu_throughput_tok_sec": round(sum(throughputs) / len(throughputs), 2),
        "total_throughput_tok_sec": round(sum(throughputs), 2),
        "total_steps": total_steps,
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    records = []

    port_counter = 0
    for num_gpus in NUM_GPUS_LIST:
        for stage in ZERO_STAGES:
            ret = run_experiment(num_gpus, stage, port=29500 + port_counter)
            port_counter += 1
            if ret != 0:
                print(f"[错误] 实验失败: GPUs={num_gpus}, Stage={stage}")
                continue

            result = collect_results(num_gpus, stage)
            if result:
                records.append(result)
                print(
                    f"[结果] GPUs={num_gpus}, Stage={stage}: "
                    f"平均显存={result['avg_peak_memory_gb']}GB, "
                    f"总吞吐量={result['total_throughput_tok_sec']}tok/s, "
                    f"单卡吞吐量={result['per_gpu_throughput_tok_sec']}tok/s"
                )
            else:
                print(f"[警告] 未找到结果文件: GPUs={num_gpus}, Stage={stage}")

    if not records:
        print("没有收集到任何有效结果。")
        sys.exit(0)

    df = pd.DataFrame(records)
    df = df.sort_values(by=["num_gpus", "zero_stage"])

    csv_path = os.path.join(RESULT_DIR, "benchmark_summary.csv")
    md_path = os.path.join(RESULT_DIR, "benchmark_summary.md")

    df.to_csv(csv_path, index=False)
    print(f"\nCSV 结果已保存: {csv_path}")

    # 同时生成 Markdown 表格
    md_lines = ["# Benchmark 结果汇总\n"]
    md_lines.append("## 显存占用与训练吞吐量\n")
    md_lines.append(
        "| 并发数 (GPUs) | ZeRO Stage | 平均显存 (GB) | 最大显存 (GB) | 单卡吞吐量 (tok/s) | 总吞吐量 (tok/s) |\n"
    )
    md_lines.append("|---|---|---|---|---|---|\n")
    for _, row in df.iterrows():
        md_lines.append(
            f"| {int(row['num_gpus'])} | {int(row['zero_stage'])} | "
            f"{row['avg_peak_memory_gb']} | {row['max_peak_memory_gb']} | "
            f"{row['per_gpu_throughput_tok_sec']} | {row['total_throughput_tok_sec']} |\n"
        )

    with open(md_path, "w") as f:
        f.writelines(md_lines)
    print(f"Markdown 结果已保存: {md_path}")

    print("\n" + "=" * 60)
    print("最终汇总表格:")
    print("=" * 60)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
