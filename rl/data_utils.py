import os
from datasets import load_dataset
from transformers import AutoTokenizer


def get_tokenizer(model_name, cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_hh_rlhf(split="train", cache_dir=None):
    ds = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    return ds


def build_rm_dataset(dataset, tokenizer, max_length=512):
    def tokenize_fn(examples):
        chosen = examples["chosen"]
        rejected = examples["rejected"]

        chosen_tokens = tokenizer(
            chosen,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        rejected_tokens = tokenizer(
            rejected,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        return {
            "input_ids_chosen": chosen_tokens["input_ids"],
            "attention_mask_chosen": chosen_tokens["attention_mask"],
            "input_ids_rejected": rejected_tokens["input_ids"],
            "attention_mask_rejected": rejected_tokens["attention_mask"],
        }

    ds = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    ds.set_format(type="torch")
    return ds


def build_ppo_dataset(dataset, tokenizer, max_length=256):
    assistant_markers = ["\n\nAssistant:", "\nAssistant:", "Assistant:"]

    def extract_prompt(text):
        for marker in assistant_markers:
            idx = text.rfind(marker)
            if idx != -1:
                return text[: idx + len(marker)]
        lines = text.split("\n")
        return "\n".join(lines[: max(1, len(lines) // 2)])

    def process_fn(examples):
        prompts = [extract_prompt(c) for c in examples["chosen"]]
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {
            "query": prompts,
            "input_ids": tokenized["input_ids"],
        }

    ds = dataset.map(process_fn, batched=True, remove_columns=dataset.column_names)
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
    ds.set_format(type="torch")
    return ds
