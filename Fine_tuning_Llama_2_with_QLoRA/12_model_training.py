"""
Notebook : Fine_tuning_Llama_2_with_QLoRA
Section  : **Model Training**
Index    : 12
"""

# ---
# ## **Model Training**
# ---

# ---
# Beyond the LoRA parameters, we also need to select optimal training arguments that enable efficient training and logging (these arguments can be left untouched, unless results are suboptimal).
#
# Note that we are training here for 3 epochs (more epochs are better but need more GPU-time).
# ---

training_arguments = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3, # Maximum gradient normal (gradient clipping)
    max_steps=-1,
    warmup_ratio=.03,
    group_by_length=True, # Group sequences into batches with same length
    lr_scheduler_type="cosine"
)

# ---
# Finally, we define an instance of the `SFTTrainer` using the base model and the PEFT configuration defined above (using `LoRAConfig`). The trainer uses these components to assemble the adapter and the training arguments to estimate the weights of the adapter. To estimate loss, we also pass in the training and validation datasets, along with the tokenizer.
# ---

trainer = SFTTrainer(
    model,
    train_dataset=formatted_training_dataset,
    eval_dataset=formatted_validation_dataset,
    peft_config=lora_config,
    dataset_text_field="formatted_prompt",
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False # Pack multiple short examples in the same input sequence to increase efficiency
)

# ---
# We can now start the training process and inspect the validation losses.
# ---

trainer.train()

# ---
# We repeat this training process with different values of $r$ and choose $r$ with the lowest validation loss.
# ---
