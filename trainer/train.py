import torch
import os
import os.path as osp
import numpy as np
import pandas as pd
from modelscope import AutoTokenizer
import json
import argparse
import time
from dataset import medicalDataset
from models import MedicalModel
from dataclasses import dataclass, field
from typing import Any, List

import torch.nn as nn
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import collate_fn
import deepspeed
from torch.utils.data.distributed import DistributedSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Trainer:
    model_engine: Any
    train_loader: Any
    val_loader: Any
    device: str
    args: Any
    criterion: Any = nn.CrossEntropyLoss()
    step_losses: List = field(default_factory=list)
    global_step: int = 0

    def train(self):
        is_main_process = self.args.local_rank in [-1,0] # single gpu or multiple gpus

        # 开始 Epoch 循环
        for epoch in range(self.args.epochs):

            if is_main_process:
                print(f"\n======== Epoch {epoch+1} / {self.args.epochs} ========")

            if is_main_process != -1:
                self.train_loader.sampler.set_epoch(epoch)
                # using tqdm
            progress_bar = tqdm(self.train_loader, desc="Training",disable=not is_main_process)
            if self.args.max_steps > 0:
                from itertools import islice
                progress_bar = tqdm(islice(self.train_loader, self.args.max_steps), total=self.args.max_steps, desc="Training", disable=not is_main_process)

            loss = self._train_epoch(progress_bar)
            if is_main_process:
                print(f"平均训练损失 (Train Loss): {loss:.4f}")

            self.evaluate(is_main_process=is_main_process)

        if is_main_process:
            self.plot_step_loss()

    def _train_epoch(self, progress_bar):
        self.model_engine.train()
        total_train_loss = 0

        device = torch.device(
            f"cuda:{self.args.local_rank}" if self.args.local_rank != -1 else "cuda"
        )

        torch.cuda.reset_peak_memory_stats(device)
        step_times = []
        tokens_per_step = []

        for batch in progress_bar:
            step_start = time.perf_counter()
            # to cuda
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # forward
            logits = self.model_engine(input_ids, mask=attention_mask)[0]

            # loss computation
            loss = self.criterion(logits, labels)
            total_train_loss += loss.item()

            # backward
            self.model_engine.backward(loss)
            self.model_engine.step()

            step_end = time.perf_counter()
            step_times.append(step_end - step_start)
            tokens_per_step.append(int(input_ids.numel()))

            self.global_step += 1
            self.step_losses.append((self.global_step, loss.item()))

            if not progress_bar.disable:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        avg_throughput = sum(t / s for t, s in zip(tokens_per_step, step_times)) / len(step_times) if step_times else 0.0

        # 保存 benchmark 指标（若处于 benchmark 模式）
        bench_gpus = os.environ.get("BENCHMARK_GPUS")
        bench_stage = os.environ.get("BENCHMARK_STAGE")
        if bench_gpus is not None and bench_stage is not None:
            os.makedirs("benchmark_results", exist_ok=True)
            rank = self.args.local_rank if self.args.local_rank != -1 else 0
            result_file = f"benchmark_results/result_gpus{bench_gpus}_stage{bench_stage}_rank{rank}.json"
            result = {
                "rank": rank,
                "peak_memory_gb": peak_memory_gb,
                "avg_throughput_tok_per_sec": avg_throughput,
                "num_steps": len(step_times),
            }
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

        return total_train_loss / len(progress_bar)

    def evaluate(self,is_main_process):
        self.model_engine.eval()
        correct_predictions = 0
        total_predictions = 0
        device = torch.device(
            f"cuda:{self.args.local_rank}" if self.args.local_rank != -1 else "cuda"
        )

        with torch.no_grad():
            val_bar = tqdm(
                self.val_loader, desc="Evaluating", disable=not is_main_process
            )
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = self.model_engine(input_ids, mask=attention_mask)[0]

                _, preds = torch.max(logits, dim=1)

                correct_predictions += torch.sum(preds == labels).item()
                total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions
        if is_main_process:
            print(f"测试集准确率 (Validation Accuracy): {accuracy * 100:.2f}%")

    def plot_step_loss(self):
        if len(self.step_losses) == 0:
            return
        steps, losses = zip(*self.step_losses)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, losses, linewidth=1.2, color="#1f77b4")
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Step-Loss Curve", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.6)
        os.makedirs("figs", exist_ok=True)
        out_path = osp.join("figs", "step_loss.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Step-Loss 图已保存至: {out_path}")
