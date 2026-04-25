"""
Ref: https://github.com/turturturturtur/ImageBindDC
(Also written by me, so some parts are reused directly.)
"""

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
from functools import partial

import torch.nn as nn
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import collate_fn
import deepspeed
from torch.utils.data.distributed import DistributedSampler
from trainer import Trainer


def main(args):
    file_path = args.file_path
    model_name = args.model_name

    train_dataset = medicalDataset("train", file_path)
    test_dataset = medicalDataset("test", file_path)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)

    print("Building DataLoader...")
    train_dataset = train_dataset.tokenize(tokenizer=tokenizer)
    test_dataset = test_dataset.tokenize(tokenizer=tokenizer)

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

    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True,
        num_workers=16,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        shuffle=False,
        pin_memory=True,
        num_workers=16,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
    )

    trainer = Trainer(
        model_engine=model_engine,
        train_loader=train_loader,
        val_loader=test_loader,
        device="cuda",
        args=args,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical Text Classification Training Script"
    )
    parser.add_argument(
        "--file_path", type=str, default="Medical_Text", help="Dataset folder path"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Pretrained model name for tokenizer",
    )
    parser.add_argument(
        "--d_model", type=int, default=1024, help="Transformer embedding dimension"
    )
    parser.add_argument("--d_hidden", type=int, default=2048, help="FFN hidden dimension")
    parser.add_argument(
        "--num_head",
        type=int,
        default=16,
        help="Number of attention heads (must divide d_model)",
    )
    parser.add_argument(
        "--num_layer", type=int, default=32, help="Number of encoder layers"
    )
    parser.add_argument("--max_len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps per epoch, -1 = no limit")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    main(args)
