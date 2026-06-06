"""
Fine-tuning Llama 2 with QLoRA — Concepts
Why fine-tune, what LoRA is, and the maths behind parameter reduction.
"""

# ── Why fine-tune? ────────────────────────────────────────────────────────────
# Full fine-tuning updates all 7B parameters of Llama 2 — prohibitively
# expensive and risks catastrophic forgetting of pre-training knowledge.
# LoRA (Low-Rank Adaptation) freezes the base weights and trains only two
# small low-rank adapter matrices per transformer block.

# ── Parameter comparison ──────────────────────────────────────────────────────
D_MODEL = 4096   # Llama 2 7B hidden dimension
R = 8            # LoRA rank

full_params   = D_MODEL * D_MODEL
lora_params   = D_MODEL * R + R * D_MODEL
reduction_pct = (1 - lora_params / full_params) * 100

print(f"Full weight matrix : {full_params:,} parameters")
print(f"LoRA adapter (r={R}): {lora_params:,} parameters")
print(f"Parameter reduction: {reduction_pct:.1f}%")

# ── QLoRA on top of LoRA ──────────────────────────────────────────────────────
# QLoRA adds 4-bit NormalFloat (NF4) quantisation of the frozen base weights,
# cutting GPU memory from ~28 GB (fp16) to ~6 GB — enabling fine-tuning on
# a single consumer GPU.

if __name__ == "__main__":
    print("\nWith QLoRA you can fine-tune Llama 2 7B on a single 16 GB GPU.")
