"""
Notebook : Fine_tuning_Llama_2_with_QLoRA
Section  : **Preparing the training data**
Index    : 09
"""

# ---
# ## **Preparing the training data**
# ---

# ---
# The first step in fine-tuning is to prepare the dataset in accordance with the prompt template of Llama 2.
#
# In this example, we reformat a [conversation-summary dataset](https://huggingface.co/datasets/knkarthick/dialogsum) to adapt it to the Llama 2 prompt pattern.
# ---

dataset = load_dataset("knkarthick/dialogsum")

# ---
# For fine-tuning, we select a random sample of 100 training examples and 50 validation examples from this dataset.
# ---

train_size, validation_size = 100, 50
training_dataset = dataset['train'].shuffle(seed=42).select(range(train_size))
validation_dataset = dataset['validation'].shuffle(seed=42).select(range(validation_size))

# ---
# The prompt format for Llama 2 is presented below:
# ---

llama2_template = """<s>[INST]<<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST]
{assistant_message}</s>"""

# ---
# We now create a formatting function that takes in an example from the random sample, and formats the example to be a `training_input` in the template format of Llama 2.
# ---

def format_input(example, prompt_template):
    system_message = "Summarize the following conversation."
    example_dialogue = example['dialogue']
    example_summary = example['summary']

    formatted_prompt = prompt_template.format(
        system_message=system_message,
        user_message=example_dialogue,
        assistant_message=example_summary
    )

    return {'formatted_prompt': formatted_prompt}

# ---
# This function can now be applied to all the instances in the training dataset and the validation dataset.
# ---

formatted_training_dataset = training_dataset.map(
    format_input,
    fn_kwargs={'prompt_template': llama2_template}
)

formatted_training_dataset[0]

formatted_validation_dataset = validation_dataset.map(
    format_input,
    fn_kwargs={'prompt_template': llama2_template}
)

formatted_validation_dataset[0]

# ---
# We will use this validation set to estimate the loss during training.
# ---
