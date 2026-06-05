"""
Notebook : Fine_tuning_Llama_2_with_QLoRA
Section  : **Saving the Trained Model**
Index    : 14
"""

# ---
# ## **Saving the Trained Model**
# ---

lora_adapter = 'llama2-7b-conversation-summarizer'

trainer.model.save_pretrained(lora_adapter)

# shell: ls -lh {lora_adapter}

# ---
# Note that only the adapter is saved, while keeping the original model intact.
#
#
# As a final step, we save this adapter to Google Drive so we can reload the adapter for inference.
# ---

from google.colab import drive
drive.mount('/content/drive')

# shell: cp -r {lora_adapter} /content/drive/MyDrive/


if __name__ == "__main__":
    pass  # entry point
