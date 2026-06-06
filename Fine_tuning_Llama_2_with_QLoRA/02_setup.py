"""
Fine-tuning Llama 2 with QLoRA — Setup & Imports
Requires a CUDA GPU (≥16 GB VRAM recommended).
"""

# requires: accelerate==0.21.0
# requires: peft==0.4.0
# requires: bitsandbytes==0.40.2
# requires: transformers==4.31.0
# requires: trl==0.4.7
# requires: datasets==2.13.0

import os
import locale
import torch

from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


def patch_locale():
    """Fix locale issue on Google Colab."""
    locale.getpreferredencoding = lambda do_setlocale=True: "UTF-8"


BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf"

# 4-bit QLoRA quantisation config
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

if __name__ == "__main__":
    patch_locale()
    print("CUDA available:", torch.cuda.is_available())
    print("Base model:", BASE_MODEL)
