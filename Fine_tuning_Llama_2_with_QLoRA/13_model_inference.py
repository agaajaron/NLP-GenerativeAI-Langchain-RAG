"""
Notebook : Fine_tuning_Llama_2_with_QLoRA
Section  : **Model Inference**
Index    : 13
"""

# ---
# ## **Model Inference**
# ---

# ---
# Once we arrive at the optimal value of $r$, we can use the best model to check the performance of the model during inference.
# ---

test_dataset = dataset['test']

test_dataset[0]

system_message = "Summarize the following conversation."
test_dialogue = test_dataset[0]['dialogue']
test_summary = test_dataset[0]['summary']

prompt_template = """<s>[INST]<<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST]"""

prompt = prompt_template.format(
    system_message=system_message,
    user_message=test_dialogue
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

generation_output = trainer.model.generate(
      input_ids=input_ids, max_new_tokens=128
)

tokenizer.decode(generation_output[0])

test_summary
