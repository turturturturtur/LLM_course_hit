"""
Visualization script for:
  Task 2: Tensor dimension flow through Multi-Head Attention
  Task 3: Attention heatmap on a sample medical text
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from models.model import MedicalModel

# ──────────────────────────────────────────────────────────────────────────────
# Task 2: Dimension Flow Diagram
# ──────────────────────────────────────────────────────────────────────────────

def draw_dimension_flow(batch=2, seq_len=128, d_model=512, num_head=8):
    d_k = d_model // num_head

    stages = [
        ("Input x\n(Before Attention)", f"[{batch}, {seq_len}, {d_model}]",  "#AED6F1"),
        ("Linear Projection Q/K/V\n(w_q / w_k / w_v)", f"[{batch}, {seq_len}, {d_model}]",  "#AED6F1"),
        ("view  →  Split Heads\n(reshape)", f"[{batch}, {seq_len}, {num_head}, {d_k}]",  "#A9DFBF"),
        ("transpose(1,2)\n(swap seq_len & num_head)", f"[{batch}, {num_head}, {seq_len}, {d_k}]",  "#A9DFBF"),
        ("Scaled Dot-Product\nAttention", f"[{batch}, {num_head}, {seq_len}, {d_k}]",  "#F9E79F"),
        ("transpose(1,2)\n(swap back)", f"[{batch}, {seq_len}, {num_head}, {d_k}]",  "#F0B27A"),
        ("contiguous + view\n(Concat all heads)", f"[{batch}, {seq_len}, {d_model}]",  "#F1948A"),
        ("w_o Linear Projection\n(Output)", f"[{batch}, {seq_len}, {d_model}]",  "#D2B4DE"),
    ]

    fig, ax = plt.subplots(figsize=(7, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(stages) * 2 + 1)
    ax.axis("off")

    box_w, box_h = 7, 1.3
    x0 = 1.5

    for i, (label, shape, color) in enumerate(stages):
        y = (len(stages) - 1 - i) * 2 + 0.5
        rect = mpatches.FancyBboxPatch(
            (x0, y), box_w, box_h,
            boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor="#555", facecolor=color
        )
        ax.add_patch(rect)
        ax.text(x0 + box_w / 2, y + box_h * 0.62, label,
                ha="center", va="center", fontsize=9)
        ax.text(x0 + box_w / 2, y + box_h * 0.22, shape,
                ha="center", va="center", fontsize=9,
                color="#1a1a8c", fontweight="bold", family="monospace")

        if i < len(stages) - 1:
            arrow_y_start = y + box_h
            arrow_y_end   = y + box_h + (2 - box_h) - 0.05
            ax.annotate(
                "", xy=(x0 + box_w / 2, arrow_y_end),
                xytext=(x0 + box_w / 2, arrow_y_start),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5)
            )

    fig.suptitle(
        f"Multi-Head Attention: Tensor Dimension Flow\n"
        f"(batch={batch}, seq_len={seq_len}, d_model={d_model}, num_head={num_head}, d_k={d_k})",
        fontsize=11, y=0.98
    )

    out = "dimension_flow.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Task 2] Saved → {out}")


def _has_simhei():
    """Check if SimHei font is available for Chinese rendering."""
    try:
        import matplotlib.font_manager as fm
        fonts = [f.name for f in fm.fontManager.ttflist]
        return "SimHei" in fonts
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Task 3: Attention Heatmap
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "The patient presents with acute chest pain radiating to the left arm, "
    "diaphoresis and shortness of breath. ECG shows ST-segment elevation "
    "consistent with myocardial infarction. Troponin levels are significantly elevated."
)

def draw_attention_heatmap(max_display_tokens=30):
    # Simple whitespace tokenizer – no external model needed
    tokens = SAMPLE_TEXT.split()[:max_display_tokens]
    seq_len = len(tokens)

    # Small model for CPU inference
    d_model, num_head, num_layer = 64, 4, 2
    vocab_size = 1000
    model = MedicalModel(
        vocab_size=vocab_size, d_model=d_model, d_hidden=256,
        max_len=512, num_classes=5, num_head=num_head,
        num_layer=num_layer, dropout=0.0
    )
    model.eval()

    # Fake token ids (just use word-hash mod vocab_size)
    input_ids = torch.tensor(
        [[hash(t) % vocab_size for t in tokens]], dtype=torch.long
    )  # [1, seq_len]

    with torch.no_grad():
        _, all_attn_weights = model(input_ids)

    # all_attn_weights[0]: [batch=1, num_head, seq_len, seq_len]
    attn = all_attn_weights[0][0]          # [num_head, seq_len, seq_len]
    attn_avg = attn.mean(0).numpy()        # [seq_len, seq_len]

    fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.45), max(6, seq_len * 0.38)))
    im = ax.imshow(attn_avg, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tokens, fontsize=7)

    ax.set_xlabel("Key tokens", fontsize=10)
    ax.set_ylabel("Query tokens", fontsize=10)
    ax.set_title(
        "Medical Text Attention Heatmap\n"
        "(Layer 1, avg over all heads)\n"
        f'"{SAMPLE_TEXT[:60]}..."',
        fontsize=9
    )

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    plt.tight_layout()

    out = "attention_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Task 3] Saved → {out}")


if __name__ == "__main__":
    print("=== Task 2: Drawing dimension flow diagram ===")
    draw_dimension_flow()

    print("=== Task 3: Drawing attention heatmap ===")
    draw_attention_heatmap()

    print("Done.")
