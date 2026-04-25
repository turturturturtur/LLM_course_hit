from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from trl import SFTTrainer, SFTConfig
import os.path as osp
from datasets import Dataset
import json


def load_label_map(path: str) -> dict:
    """Load label.json, format: {"0": "Neoplasms", ...}"""
    label_path = osp.join(path, "label.json")
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_medical_data(path: str, split: str, label_map: dict):
    """Parse medical text data into dicts with text / category / label."""
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
                        label_idx = int(parts[0]) - 1  # file uses 1-5, convert to 0-4
                        text = parts[1]
                        category = label_map.get(str(label_idx), "Unknown")
                        yield {"text": text, "category": category, "label": label_idx}
                else:
                    yield {"text": line, "category": "Unknown", "label": 0}
            except Exception:
                continue


def apply_chat_template(example, tokenizer):
    """Convert sample to instruction format and render with Qwen3 chat template."""
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

    # 1. Load label map
    label_map = load_label_map(file_path)
    print("Label map:", label_map)

    # 2. Load tokenizer (Qwen3 needs trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
    )

    # 4. Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 6. Inject LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 7. Build dataset
    train_dataset = Dataset.from_generator(
        parse_medical_data,
        gen_kwargs={"path": file_path, "split": "train", "label_map": label_map},
    )

    # 8. Apply chat template (SFTTrainer only needs the text field)
    train_dataset = train_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )

    # 9. Training args (SFTConfig replaces TrainingArguments)
    training_args = SFTConfig(
        output_dir="./output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=200,
        save_strategy="steps",
        save_steps=100,
        dataset_text_field="text",
        max_length=512,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 10. Init SFTTrainer (trl 1.2.0 uses processing_class for tokenizer)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    # 11. Train
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical Text Classification with Instruction Tuning (LoRA)"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="Medical_Text",
        help="Dataset folder path",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=".cache/Qwen3-8B",
        help="Local pretrained model path",
    )
    args = parser.parse_args()
    main(args)
