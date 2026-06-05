"""
Notebook : Fine_tuning_Llama_2_with_QLoRA
Section  : **Importing the Libraries Required**
Index    : 08
"""

# ---
# ## **Importing the Libraries Required**
# ---

import os
import torch
import locale

from datasets import load_dataset

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# ---
# The following code allows us to run shell commands on Colab (we need shell commands to check file sizes of the estimated models).
# ---

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"

locale.getpreferredencoding = getpreferredencoding
