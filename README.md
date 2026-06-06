# NLP · Generative AI · LangChain · RAG

A portfolio of LangChain projects covering EDA agents, fine-tuning, RAG pipelines, and conversational memory — all using modern LangChain ≥ 0.2 patterns (LCEL, `langchain_openai`, `langchain_community`).

## Projects

### 1. Data Science ML Assistants with LangChain
[![LangChain](https://img.shields.io/badge/LangChain-0.2-blue)](Developing_datascience_ML_Assistants_with_LLM_LangChain/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green)](Developing_datascience_ML_Assistants_with_LLM_LangChain/)

Pandas DataFrame agent answers natural-language EDA and ML questions on the SA Heart Disease dataset. Covers exploratory statistics, visualisation, hypothesis testing, and multi-step classifier training (Logistic Regression, Random Forest, Neural Net).

```
User prompt → Pandas EDA Agent (GPT-4) → code execution → answer
```

→ [`Developing_datascience_ML_Assistants_with_LLM_LangChain/`](Developing_datascience_ML_Assistants_with_LLM_LangChain/)

---

### 2. Fine-tuning Llama 2 with QLoRA
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](Fine_tuning_Llama_2_with_QLoRA/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)](Fine_tuning_Llama_2_with_QLoRA/)
[![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-4--bit-red)](Fine_tuning_Llama_2_with_QLoRA/)

Instruction fine-tuning of Llama 2 7B on DialogSum using QLoRA (4-bit NF4 quantisation + LoRA adapters). Covers dataset preparation, SFTTrainer training loop, inference, and adapter saving.

```
DialogSum (train/val)
    ↓  Llama-2 chat template
BitsAndBytesConfig (4-bit NF4)
    ↓  LoRA adapters (r=8, alpha=16)
SFTTrainer → llama2-7b-conversation-summarizer
```

→ [`Fine_tuning_Llama_2_with_QLoRA/`](Fine_tuning_Llama_2_with_QLoRA/)

---

### 3. LangChain RAG Data Science Assistant
[![LangChain](https://img.shields.io/badge/LangChain-0.2-blue)](NLP_Final_Project_LangChain_RAG_Data_Science_Assistant/)
[![FAISS](https://img.shields.io/badge/FAISS-vector--store-orange)](NLP_Final_Project_LangChain_RAG_Data_Science_Assistant/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-red)](NLP_Final_Project_LangChain_RAG_Data_Science_Assistant/)

Four agent types in one project: Pandas EDA agent, NL-driven ML modeling, SQL agent over SQLite, and a conversational RAG pipeline (PDF → FAISS → retriever tool → SerpAPI fallback).

```
PDF → PyPDFLoader → chunks → OpenAIEmbeddings → FAISS
                                                    ↓  similarity_search(k=4)
                                        GPT-4 ← relevant chunks
                                                    ↓  (out-of-scope → SerpAPI)
```

→ [`NLP_Final_Project_LangChain_RAG_Data_Science_Assistant/`](NLP_Final_Project_LangChain_RAG_Data_Science_Assistant/)

---

### 4. Conversational Memory Chatbot
[![LangChain](https://img.shields.io/badge/LangChain-0.2-blue)](conversational_memory_chatbot/)
[![LCEL](https://img.shields.io/badge/LCEL-RunnableWithMessageHistory-purple)](conversational_memory_chatbot/)

Minimal chatbot built with `RunnableWithMessageHistory` and LCEL. Demonstrates session-scoped in-memory history so the model recalls earlier turns. Includes a scripted demo and an interactive REPL.

```
User message → ChatPromptTemplate (system + history + input)
                        ↓  LCEL (PROMPT | llm)
                  ChatOpenAI (GPT-4)
                        ↓
              RunnableWithMessageHistory → session store
```

→ [`conversational_memory_chatbot/`](conversational_memory_chatbot/)

---

## Quick start

```bash
git clone https://github.com/agaajaron/nlp-generativeai-langchain-rag.git
cd nlp-generativeai-langchain-rag
cp .env.example .env          # fill in OPENAI_API_KEY (and optionally SERPAPI_API_KEY)

# Run any project
cd conversational_memory_chatbot
pip install -r requirements.txt
python 02_chatbot.py
```

## Repository layout

```
.
├── Developing_datascience_ML_Assistants_with_LLM_LangChain/
│   ├── 01_setup.py
│   ├── 02_eda_queries.py
│   ├── 03_visualisation.py
│   └── 04_ml_modeling.py
├── Fine_tuning_Llama_2_with_QLoRA/
│   ├── 01_concepts.py
│   ├── 02_setup.py
│   ├── 03_prepare_data.py
│   ├── 04_train_model.py
│   └── 05_inference_and_save.py
├── NLP_Final_Project_LangChain_RAG_Data_Science_Assistant/
│   ├── 01_setup.py
│   ├── 02_eda_agent.py
│   ├── 03_ml_modeling.py
│   ├── 04_sql_agent.py
│   └── 05_rag_pipeline.py
├── conversational_memory_chatbot/
│   ├── 01_setup.py
│   └── 02_chatbot.py
├── notebooks/originals/      # archived original .ipynb files
└── .env.example
```

## Tech stack

| Library | Version | Used for |
|---------|---------|----------|
| `langchain` | ≥ 0.2 | Agent orchestration, LCEL |
| `langchain-openai` | ≥ 0.1 | ChatOpenAI, OpenAIEmbeddings |
| `langchain-community` | ≥ 0.2 | FAISS, SQL toolkit, document loaders |
| `langchain-experimental` | ≥ 0.0.60 | Pandas DataFrame agent |
| `faiss-cpu` | ≥ 1.7 | Vector store for RAG |
| `transformers` / `peft` / `trl` | pinned | QLoRA fine-tuning |
| `tensorflow` | ≥ 2.13 | Neural network classifier |
