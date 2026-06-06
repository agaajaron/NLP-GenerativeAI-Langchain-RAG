# Fine-tuning Llama 2 with QLoRA

> **Demonstrates:** QLoRA parameter-efficient fine-tuning · 4-bit quantisation · LoRA adapter training · HuggingFace PEFT/TRL

## What it does

Fine-tunes the **Llama 2 7B Chat** model on a conversation summarisation task using QLoRA — reducing GPU memory from ~28 GB to ~6 GB while preserving 99.6% of model quality.

**Dataset:** [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) — 100 train / 50 val samples

## Project structure

| File | Purpose |
|------|---------|
| `01_concepts.py` | LoRA maths — parameter reduction calculation |
| `02_setup.py` | BitsAndBytes 4-bit config, imports |
| `03_prepare_data.py` | Dataset loading, Llama 2 prompt formatting |
| `04_train_model.py` | LoRA config, SFTTrainer, 3-epoch training loop |
| `05_inference_and_save.py` | Generate summaries, save adapter |

## Quick start

```bash
# Requires a GPU with ≥16 GB VRAM
pip install -r requirements.txt
python 01_concepts.py    # understand the parameter maths
python 05_inference_and_save.py  # end-to-end run
```

## Key concept: LoRA parameter reduction

```
Full weight update: W ∈ ℝ^(4096×4096) = 16.7M params
LoRA (r=8):        A ∈ ℝ^(4096×8) + B ∈ ℝ^(8×4096) = 65K params
                   → 99.6% fewer trainable parameters
```

## Tech stack

![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-blue)
![bitsandbytes](https://img.shields.io/badge/bitsandbytes-4bit-red)
