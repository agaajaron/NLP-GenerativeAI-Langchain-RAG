"""
Fine-tuning Llama 2 with QLoRA — Model Loading, LoRA Config & Training
Attaches LoRA adapters to every transformer block and trains for 3 epochs.
"""

import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

from setup import BASE_MODEL, BNB_CONFIG
from prepare_data import load_datasets

LORA_CONFIG = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "down_proj", "up_proj", "lm_head"],
    task_type="CAUSAL_LM",
)

TRAINING_ARGS = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=1e-3,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_steps=0,
)


def load_model_and_tokenizer():
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = LlamaTokenizerFast.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def train():
    train_ds, val_ds = load_datasets()
    model, tokenizer = load_model_and_tokenizer()

    trainer = SFTTrainer(
        model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=LORA_CONFIG,
        dataset_text_field="formatted_prompt",
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=TRAINING_ARGS,
        packing=False,
    )

    trainer.train()
    return trainer


if __name__ == "__main__":
    trainer = train()
    print("Training complete. Final eval loss:", trainer.state.log_history[-1])
