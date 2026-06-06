"""
Fine-tuning Llama 2 with QLoRA — Data Preparation
Loads the DialogSum dataset and formats it into Llama 2 prompt style.
"""

from datasets import load_dataset
from setup import patch_locale

patch_locale()

LLAMA2_TEMPLATE = """<s>[INST]<<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST]
{assistant_message}</s>"""

SYSTEM_MESSAGE = "Summarize the following conversation."

TRAIN_SIZE = 100
VAL_SIZE   = 50


def format_example(example, prompt_template=LLAMA2_TEMPLATE):
    return {
        "formatted_prompt": prompt_template.format(
            system_message=SYSTEM_MESSAGE,
            user_message=example["dialogue"],
            assistant_message=example["summary"],
        )
    }


def load_datasets():
    dataset = load_dataset("knkarthick/dialogsum")
    train = dataset["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
    val   = dataset["validation"].shuffle(seed=42).select(range(VAL_SIZE))
    train = train.map(format_example)
    val   = val.map(format_example)
    return train, val


if __name__ == "__main__":
    train_ds, val_ds = load_datasets()
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print("\nSample formatted prompt:\n")
    print(train_ds[0]["formatted_prompt"][:400])
