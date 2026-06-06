"""
Fine-tuning Llama 2 with QLoRA — Inference & Saving
Runs the fine-tuned model on a test example and saves the LoRA adapter.
"""

import torch
from prepare_data import load_datasets, LLAMA2_TEMPLATE, SYSTEM_MESSAGE
from train_model import load_model_and_tokenizer

ADAPTER_NAME = "llama2-7b-conversation-summarizer"

INFERENCE_TEMPLATE = """<s>[INST]<<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST]"""


def summarise(trainer, tokenizer, dialogue: str, max_new_tokens: int = 128) -> str:
    prompt = INFERENCE_TEMPLATE.format(
        system_message=SYSTEM_MESSAGE,
        user_message=dialogue,
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    output = trainer.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    from train_model import train
    from datasets import load_dataset

    trainer = train()
    _, tokenizer = load_model_and_tokenizer()

    test_example = load_dataset("knkarthick/dialogsum")["test"][0]
    print("Reference summary:\n", test_example["summary"])
    print("\nModel summary:\n", summarise(trainer, tokenizer, test_example["dialogue"]))

    # Save only the lightweight adapter (not the full 7B base model)
    trainer.model.save_pretrained(ADAPTER_NAME)
    print(f"\nAdapter saved to ./{ADAPTER_NAME}/")
