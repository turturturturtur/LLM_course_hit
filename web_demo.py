"""
Gradio Web Demo: Medical Text Classification Model Comparison
Compare inference results between LoRA SFT and Full SFT (Base SFT).
"""

import gradio as gr
import torch
import json
import os.path as osp
import random
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ====================== Config ======================
BASE_MODEL_PATH = ".cache/Qwen3-8B"
LORA_ADAPTER_PATH = "output/checkpoint-200"
FULL_SFT_PATH = "output_base/checkpoint-200"
DATA_PATH = "Medical_Text"

LABEL_MAP = {
    "0": "Neoplasms",
    "1": "Digestive System Diseases",
    "2": "Nervous System Diseases",
    "3": "Cardiovascular Diseases",
    "4": "General Pathological Conditions",
}
LABEL_LIST = [LABEL_MAP[str(i)] for i in range(5)]

MAX_NEW_TOKENS = 128
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ====================== Model Loading ======================
print("[1/3] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

print("[2/3] Loading LoRA SFT model...")
base_model_lora = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    quantization_config=bnb_config,
    device_map={"": DEVICE},
    dtype=torch.float16,
)
lora_model = PeftModel.from_pretrained(base_model_lora, LORA_ADAPTER_PATH)
lora_model.eval()

print("[3/3] Loading Full SFT model...")
full_model = AutoModelForCausalLM.from_pretrained(
    FULL_SFT_PATH,
    local_files_only=True,
    quantization_config=bnb_config,
    device_map={"": DEVICE},
    dtype=torch.float16,
)
full_model.eval()

print("All models loaded!")


# ====================== Inference ======================
def build_messages(text: str):
    """Build chat template messages consistent with training."""
    return [
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
    ]


def parse_think_content(text: str) -> tuple:
    """Parse Qwen3 <think> tags, return (think_content, final_answer)."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        final_answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return think_content, final_answer
    return "", text


def extract_predicted_label(text: str) -> str:
    """Extract predicted label from generated text."""
    for label in LABEL_LIST:
        if label.lower() in text.lower():
            return label
    return "Unknown"


@torch.no_grad()
def predict(model, text: str) -> dict:
    """Run inference on a single model, return predicted label and generated text."""
    messages = build_messages(text)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    think_content, final_answer = parse_think_content(generated_text)
    predicted_label = extract_predicted_label(final_answer)

    return {
        "predicted_label": predicted_label,
        "think_content": think_content,
        "final_answer": final_answer,
        "raw_text": generated_text,
    }


def compare_models(text: str):
    """Run both models and return comparison results (Markdown format)."""
    if not text or not text.strip():
        return "Please enter medical text", "Please enter medical text"

    lora_result = predict(lora_model, text)
    full_result = predict(full_model, text)

    # LoRA output
    lora_think_md = f"""
<details>
<summary>Thinking process (click to expand)</summary>

{lora_result['think_content'] or '*No thinking process*'}

</details>
""" if lora_result["think_content"] else ""

    lora_output = f"""### Predicted Label: {lora_result['predicted_label']}

{lora_think_md}

### Generated Response
{lora_result['final_answer']}
"""

    # Full SFT output
    full_think_md = f"""
<details>
<summary>Thinking process (click to expand)</summary>

{full_result['think_content'] or '*No thinking process*'}

</details>
""" if full_result["think_content"] else ""

    full_output = f"""### Predicted Label: {full_result['predicted_label']}

{full_think_md}

### Generated Response
{full_result['final_answer']}
"""

    return lora_output, full_output


# ====================== Load Test Samples ======================
def load_test_samples(n=6):
    """Load sample texts from the test set."""
    filepath = osp.join(DATA_PATH, "test.dat")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    random.seed(42)
    return random.sample(lines, min(n, len(lines)))


EXAMPLES = load_test_samples(6)


# ====================== Gradio UI ======================
css = """
.model-box {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    background-color: #fafafa;
    color: #111111 !important;
}
.model-box p, .model-box h3, .model-box h4, .model-box details, .model-box summary {
    color: #111111 !important;
}
.lora-box {
    border-left: 4px solid #3b82f6;
}
.full-box {
    border-left: 4px solid #f59e0b;
}
"""

with gr.Blocks(title="Medical Text Classification Comparison") as demo:
    gr.Markdown(
        """
        # Medical Text Classification Model Comparison

        Side-by-side comparison of **LoRA SFT** and **Full SFT (Base SFT)**
        on medical text classification.

        | Model | Description |
        |------|------|
        | **LoRA SFT** | Only low-rank adapter parameters are trained; small model size, efficient training |
        | **Full SFT** | All parameters are fine-tuned; usually better performance but higher training cost |
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Medical Text")
            input_text = gr.Textbox(
                label="Medical Abstract",
                placeholder="Enter medical text here...",
                lines=10,
                value=EXAMPLES[0],
            )

            with gr.Row():
                submit_btn = gr.Button("Compare Predictions", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", size="lg")

            gr.Markdown("### Quick Examples (click to fill)")
            example_btns = []
            with gr.Row():
                for i, ex in enumerate(EXAMPLES[:3]):
                    btn = gr.Button(f"Example {i+1}", size="sm")
                    example_btns.append(btn)
            with gr.Row():
                for i, ex in enumerate(EXAMPLES[3:6], start=4):
                    btn = gr.Button(f"Example {i}", size="sm")
                    example_btns.append(btn)

        with gr.Column(scale=2):
            gr.Markdown("### Prediction Comparison")
            with gr.Row():
                with gr.Column(elem_classes=["model-box", "lora-box"]):
                    gr.Markdown("#### LoRA SFT")
                    lora_output = gr.Markdown()
                with gr.Column(elem_classes=["model-box", "full-box"]):
                    gr.Markdown("#### Full SFT (Base SFT)")
                    full_output = gr.Markdown()

    # Event bindings
    submit_btn.click(
        fn=compare_models,
        inputs=input_text,
        outputs=[lora_output, full_output],
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[lora_output, full_output],
    )

    for btn, ex in zip(example_btns, EXAMPLES):
        btn.click(fn=lambda text=ex: text, inputs=None, outputs=input_text)
        btn.click(fn=compare_models, inputs=input_text, outputs=[lora_output, full_output])

    gr.Markdown(
        """
        ---
        **Note:**
        - Models are SFT-tuned based on Qwen3-8B; output may contain `<think>` reasoning tags
        - Predicted labels are extracted via keyword matching; refer to the generated response for the final answer
        - If `<think>` is empty, the model directly produced the answer
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=17860, share=False, css=css)
