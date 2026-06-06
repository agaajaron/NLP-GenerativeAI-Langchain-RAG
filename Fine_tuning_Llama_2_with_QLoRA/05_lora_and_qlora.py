"""
Notebook : Fine_tuning_Llama_2_with_QLoRA
Section  : **LoRA and QLoRA**
Index    : 05
"""

# ---
# ## **LoRA and QLoRA**
# ---

# ---
# **Fine-tuning all the 7 billion parameters of Llama 2 (i.e., full fine-tuning)** to adapt to a specific task is not only **prohibitively expensive computationally**, but also runs the risk of the model losing the knowledge ingested during the training phase.
#
# **Instead, we use a different approach - Low Rank Adaptation (LoRA)** - that reduces the number of parameter updates that need to be done while retaining the knowledge ingested during pretraining. LoRA is one among a family of models that are referred to as Parameter-Efficient Fine Tuning (PEFT) models. [The LoRA approach](https://arxiv.org/abs/2106.09685) is presented in the figure below.
# ---

# ---
# # [embedded image removed]
# ---

# ---
# [Source](https://huggingface.co/docs/peft/conceptual_guides/lora)
# ---

# ---
# As depicted in the above figure, the original weights of the transformer blocks $W$ remain frozen during fine-tuning with LoRA. Training is confined to two low-rank ($r$) matrices $A$ and $B$ and the learned deposition of $W = BA$ is then added back to the pretrained weights. In this way, LoRA acts as a source of adapter matrices that can be trained with a much smaller computational footprint. For example, while the original model has a dimensionality of $d_{\text{model}}=4096$ (across all the 32 layers), we typically use a LoRA adaptor with $r = 8$ or $r = 16$. This dramatically reduces the parameters that need to be estimated. For example, for $r = 8$, $A \in \mathbb{R}^{4096 \times 8}$ and $A \in \mathbb{R}^{8 \times 4096}$.
#
# ---

print((4096*4096),(4096*8 + 8*4096))

(4096*8 + 8*4096)/((4096*4096))

# ---
# There are two main parameters to tune in the LoRA approach.
# - rank ($r$): usually a choice between 4, 8, or 16
# - [scaling coefficient ($\alpha$)](https://civitai.com/articles/2125/what-lora-alpha-actually-does-in-theory): usually fixed at 16
# ---

# ---
# $r$ defines the dimensions of the low-rank matrices, while $\alpha$ determines the scaling factor for the weight matrices. It is common to [freeze $\alpha = 16$](https://arxiv.org/pdf/2308.07317v1.pdf), while varying the values of $r = \alpha, \alpha/2, \alpha/4$ and arriving at the optimal value of $r$ that gives the lowest validation loss (note that we use the same loss used for the base model, e.g., perplexity or log loss).
# ---

# ---
# [QLoRA](https://arxiv.org/pdf/2305.14314.pdf) improves the efficiency of LoRA training further by optimizing few key parameters of the LoRA adapter (see figure below). Key optimizations executed by QLoRA are:
#
# - 4-bit NormalFloat quantization that shifts the base Llama2 model from 16-bit to 4-bit
# - paged optimizers that shift weights to CPU if the GPU RAM is full (instead of throwing up an error)
#
# ---

# ---
# # [embedded image removed]
# ---

# ---
# Let us now implement the fine-tuning procedure described in this section for the Llama 2 7-billion parameter chat model using QLoRA.
# ---
