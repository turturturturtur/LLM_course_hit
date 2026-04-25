import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from trl import SFTTrainer, SFTConfig
import os.path as osp
from datasets import Dataset
import json


def load_label_map(path: str) -> dict:
    """load label"""
    label_path = osp.join(path, "label.json")
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_medical_data(path: str, split: str, label_map: dict):
    """parser"""
    filename = "train.dat" if split == "train" else "test.dat"
    filepath = osp.join(path, filename)

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                if split == "train":
                    parts = line.split("\t")
                    if len(parts) == 2:
                        label_idx = int(parts[0]) - 1  # convert index from 1-5 -> 0-4
                        text = parts[1]
                        category = label_map.get(str(label_idx), "Unknown")
                        yield {"text": text, "category": category, "label": label_idx}
                else:
                    yield {"text": line, "category": "Unknown", "label": 0}
            except Exception:
                continue


def apply_chat_template(example, tokenizer):
    """chat template"""
    text = example["text"]
    category = example["category"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional medical text classification assistant. "
                "Classify the given medical abstract into one of the predefined categories."
            ),
        },
        {
            "role": "user",
            "content": (
                "Please classify the following medical text into one of these categories: "
                "Neoplasms, Digestive System Diseases, Nervous System Diseases, "
                "Cardiovascular Diseases, General Pathological Conditions.\n\n"
                f"Medical Text:\n{text}"
            ),
        },
        {"role": "assistant", "content": category},
    ]

    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": rendered}


def main(args):
    file_path = args.file_path
    model_path = args.model_path

    # load label
    label_map = load_label_map(file_path)
    print("Label map:", label_map)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load weight
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # print the trainable parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # build dataset
    train_dataset = Dataset.from_generator(
        parse_medical_data,
        gen_kwargs={"path": file_path, "split": "train", "label_map": label_map},
    )

    # apply chat template
    train_dataset = train_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )

    # SFT config(和实验setting相同)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=100,
        dataset_text_field="text",
        max_length=512,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        fp16=True,
        deepspeed=args.deepspeed,
    )

    # initialize
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    # train
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical Text Classification with Full Fine-tuning (8 GPUs)"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="Medical_Text",
        help="数据集所在的文件夹路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=".cache/Qwen3-8B",
        help="预训练模型本地路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_base",
        help="输出目录",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="每张卡的训练 batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="最大训练步数",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="ds_config_zero2.json",
        help="DeepSpeed 配置文件路径",
    )
    args = parser.parse_args()
    main(args)
