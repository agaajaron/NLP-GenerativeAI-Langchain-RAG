"""
Notebook : Fine_tuning_Llama_2_with_QLoRA
Section  : **Defining the LoRA Parameters**
Index    : 11
"""

# ---
# ## **Defining the LoRA Parameters**
# ---

# ---
# We begin by $\alpha = 16$, and a starting value of $r = 4$. If the observed loss is sub-optimal then we can increase the value of $r$ to 8 and 16. It is not a good practise to increase the value of $r > \alpha$.
# ---

lora_r = 4

lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

# Load LoRA configuration
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head'],
    task_type="CAUSAL_LM"
)

# ---
# Note that we want to attach LoRA adapters to all the components of the transformer block (this is shown to yield the best results).
# ---
