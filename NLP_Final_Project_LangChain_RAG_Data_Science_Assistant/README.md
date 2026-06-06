# LangChain RAG Data Science Assistant — Final Project

> **Demonstrates:** Pandas EDA agent · NL-driven ML modeling · SQL agent · Retrieval-Augmented Generation · Multi-tool conversational agent

## What it does

A full-stack LangChain assistant that integrates four agent types in one project:

| Agent | What it does |
|-------|-------------|
| **Pandas EDA** | Explores the SA Heart Disease dataset via natural language |
| **ML Modeling** | Runs hypothesis tests + 3 classifiers from plain-English prompts |
| **SQL Agent** | Queries a SQLite database in natural language with math tool |
| **RAG Agent** | Answers questions from a PDF; falls back to web search for out-of-scope queries |

**Dataset:** [South Africa Heart Disease](https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data)

## Project structure

| File | Purpose |
|------|---------|
| `01_setup.py` | LLM init, shared imports, agent factory |
| `02_eda_agent.py` | Dataset loading + 7 EDA queries (stats, plots, heatmap) |
| `03_ml_modeling.py` | Hypothesis test + Logistic Regression + Random Forest + Neural Net |
| `04_sql_agent.py` | SQLite DB creation + 4 NL queries with math tool |
| `05_rag_pipeline.py` | PDF → FAISS → conversational retrieval agent |

## Quick start

```bash
pip install -r requirements.txt
cp ../.env.example ../.env   # fill in your keys
python 02_eda_agent.py       # EDA
python 04_sql_agent.py       # SQL
python 05_rag_pipeline.py    # RAG
```

## Architecture: RAG pipeline

```
PDF document
    ↓  PyPDFLoader
Chunks (1000 chars, 50 overlap)
    ↓  OpenAIEmbeddings
FAISS vector store
    ↓  similarity_search(k=4)
Relevant chunks → LLM context
    ↓  GPT-4
Answer  (or → SerpAPI fallback)
```

## Tech stack

![LangChain](https://img.shields.io/badge/LangChain-0.2-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green)
![FAISS](https://img.shields.io/badge/FAISS-vector--store-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-red)
