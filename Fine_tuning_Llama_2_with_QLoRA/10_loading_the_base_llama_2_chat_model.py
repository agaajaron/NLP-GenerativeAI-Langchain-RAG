"""
Notebook : Fine_tuning_Llama_2_with_QLoRA
Section  : **Loading the Base Llama 2 Chat Model**
Index    : 10
"""

# ---
# ## **Loading the Base Llama 2 Chat Model**
# ---

# ---
# We begin the fine-tuning by downloading as our base the Llama 2 chat model. However, we will need to load it in 4-bit precision as prescribed by QLoRA. This is achieved by tweaking the parameters offered by `BitsAndBytesConfig`.
#
# ---

base_model_name = "NousResearch/Llama-2-7b-chat-hf"

# Load tokenizer and model with QLoRA configuration
# Compute dtype for 4-bit base models
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False
)

# Load the entire model on the GPU 0
device_map = {"": 0}

# ---
# We can now load the pre-trained Llama 2 model in this configuration like so:
# ---

model = LlamaForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# ---
# We also load the tokenizer specific to Llama 2 like so:
# ---

tokenizer = LlamaTokenizerFast.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
