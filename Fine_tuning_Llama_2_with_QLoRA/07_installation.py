"""
Notebook : Fine_tuning_Llama_2_with_QLoRA
Section  : **Installation**
Index    : 07
"""

# ---
# ## **Installation**
# ---

# requires: accelerate==0.21.0
# requires (cont): peft==0.4.0
# requires (cont): bitsandbytes==0.40.2
# requires (cont): transformers==4.31.0
# requires (cont): trl==0.4.7
# requires (cont): datasets==2.13.0

# ---
# The functionalities we use of these packages are:
# - `transformers`, `datasets`: helpers to load models and datsets from the HuggingFace ecosystem
# - `peft`: contains implementation of LoRA
# - `bitsandbytes`: facilitates application of QLoRA in conjunction with `peft`
# - `trl`: abstractions to train the LoRA adapters
# ---
