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


import torch
from torch.utils.data import DataLoader


def create_collate_fn(pad_token_id=0):
    def collate_fn(batch):
        # 假设你的 dataset 返回的格式是字典：{'input_ids': [101, 23, ...], 'label': 1}
        # batch 就是一个包含了 batch_size 个这样字典的列表

        # 1. 把所有的 input_ids 和 labels 抽出来
        input_ids_list = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        # 2. 找到这个 batch 里最长的句子长度
        max_len = max(len(ids) for ids in input_ids_list)

        batch_input_ids = []
        batch_attention_mask = []

        # 3. 动态 Padding 并生成 Mask
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            # 补齐 input_ids
            padded_ids = ids + [pad_token_id] * pad_len
            # 生成 mask (真实的词是 1，补齐的 PAD 是 0)
            mask = [1] * len(ids) + [0] * pad_len

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(mask)

        # 4. 转成 PyTorch 的 Tensor
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


import torch.nn as nn
from tqdm import tqdm  # 用来显示酷炫的进度条


def train_model(model, train_loader, val_loader, args):
    # 1. 准备设备 (优先使用 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备是: {device}")
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(
            f"{torch.cuda.device_count()} gpus！DataParallel..."
        )
        model = nn.DataParallel(model)

    # 2. 定义损失函数 (CrossEntropyLoss 内部自带 LogSoftmax)
    criterion = nn.CrossEntropyLoss()

    # 3. 定义优化器 (AdamW 是目前训练 Transformer 的绝对主流)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 开始 Epoch 循环
    for epoch in range(args.epochs):
        print(f"\n======== Epoch {epoch+1} / {args.epochs} ========")

        # ---------------- 训练阶段 ----------------
        model.train()  # 开启训练模式 (启用 Dropout 和 BatchNorm)
        total_train_loss = 0

        # 使用 tqdm 包装 train_loader，显示进度条
        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            # A. 把数据搬到 GPU 上
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # B. 梯度清零 (PyTorch 默认会累加梯度，所以每次必须清零)
            optimizer.zero_grad()

            # C. 前向传播
            logits = model(input_ids, mask=attention_mask)

            # D. 计算损失
            loss = criterion(logits, labels)
            total_train_loss += loss.item()

            # E. 反向传播，更新权重
            loss.backward()

            # (可选) 梯度裁剪，防止 Transformer 训练初期梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 更新进度条显示的 Loss
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"平均训练损失 (Train Loss): {avg_train_loss:.4f}")

        # ---------------- 验证/测试阶段 ----------------
        # 每一个 Epoch 结束后，在测试集上看看效果
        evaluate_model(model, val_loader, device)


# 评估函数 (不用算梯度，速度更快)
def evaluate_model(model, val_loader, device):
    model.eval()  # 开启评估模式 (关闭 Dropout)
    correct_predictions = 0
    total_predictions = 0

    # torch.no_grad() 告诉 PyTorch：这里不需要计算梯度，省内存！
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, mask=attention_mask)

            # 找到概率最大的那个类别的索引
            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"测试集准确率 (Validation Accuracy): {accuracy * 100:.2f}%")


def main(args):
    # Read from kaggle data, which can be loaded by python( pandans and numpy can't)
    file_path = args.file_path
    model_name = args.model_name

    train_dataset = medicalDataset("train",file_path)
    test_dataset = medicalDataset("test",file_path)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = train_dataset.tokenize(tokenizer=tokenizer)
    test_dataset = test_dataset.tokenize(tokenizer=tokenizer)
    vocab_size = len(tokenizer)

    model = MedicalModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        max_len=args.max_len,
        num_classes=args.num_classes,
        num_head=args.num_head,
        num_layer=args.num_layer,
        dropout=args.dropout,
    )


    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    print("正在构建 DataLoader...")
    # 实例化 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, # 训练集必须打乱！
        collate_fn=create_collate_fn(pad_token_id)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # 测试集不需要打乱
        collate_fn=create_collate_fn(pad_token_id)
    )

    print("开始训练！🚀")
    # 把模型和数据交给 Trainer
    train_model(model, train_loader, test_loader, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical Text Classification Training Script"
    )
    parser.add_argument(
        "--file_path", type=str, default="Medical_Text", help="数据集所在的文件夹路径"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="用于加载 Tokenizer 的预训练模型名称",
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Transformer 的词向量维度"
    )
    parser.add_argument("--d_hidden", type=int, default=2048, help="FFN 隐藏层的维度")
    parser.add_argument(
        "--num_head",
        type=int,
        default=8,
        help="多头注意力的头数 (必须能被 d_model 整除)",
    )
    parser.add_argument(
        "--num_layer", type=int, default=6, help="Encoder Layer 的堆叠层数"
    )
    parser.add_argument("--max_len", type=int, default=512, help="句子最大长度")
    parser.add_argument("--num_classes", type=int, default=5, help="最终分类的类别数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    args = parser.parse_args()

    main(args)
