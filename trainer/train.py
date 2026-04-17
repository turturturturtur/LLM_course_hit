import torch
import os
import os.path as osp
import numpy as np
import pandas as pd
from modelscope import AutoTokenizer
import json
import argparse
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


@dataclass
class Trainer:
    model_engine: Any
    train_loader: Any
    val_loader: Any
    device: str
    args: Any
    criterion: Any = nn.CrossEntropyLoss()

    def train(self):
        is_main_process = self.args.local_rank in [-1,0] # single gpu or multiple gpus

        # 开始 Epoch 循环
        for epoch in range(self.args.epochs):

            print(f"\n======== Epoch {epoch+1} / {self.args.epochs} ========")

            if is_main_process != -1:
                self.train_loader.sampler.set_epoch(epoch)
                # using tqdm
            progress_bar = tqdm(self.train_loader, desc="Training",disable=not is_main_process)

            loss = self._train_epoch(progress_bar)
            if is_main_process:
                print(f"平均训练损失 (Train Loss): {loss:.4f}")

            self.evaluate(is_main_process=is_main_process)

    def _train_epoch(self, progress_bar):
        self.model_engine.train()
        total_train_loss = 0

        device = torch.device(
            f"cuda:{self.args.local_rank}" if self.args.local_rank != -1 else "cuda"
        )

        for batch in progress_bar:
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

            if not progress_bar.disable:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

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
