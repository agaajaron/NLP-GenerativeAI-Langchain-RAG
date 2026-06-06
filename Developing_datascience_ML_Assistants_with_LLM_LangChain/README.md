# LangChain Data Science Assistant

> **Demonstrates:** Pandas DataFrame Agent · EDA via natural language · LLM-driven visualisation · Prompt-based ML modeling

## What it does

A GPT-4 powered agent that treats a Pandas DataFrame as its workspace. You describe what you want in plain English — the agent generates, executes, and explains the Pandas/Scikit-learn code autonomously using LangChain's ReAct framework.

**Dataset:** [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## Project structure

| File | Purpose |
|------|---------|
| `01_setup.py` | LLM init, shared `make_agent()` factory |
| `02_eda_queries.py` | Row counts, column info, missing values, summary stats |
| `03_visualisation.py` | Correlation heatmap + t-test hypothesis testing |
| `04_ml_modeling.py` | Full Logistic Regression pipeline via a single prompt |

## Quick start

```bash
pip install -r requirements.txt
cp ../.env.example ../.env   # add your API keys
python 02_eda_queries.py
```

## Key concept: ReAct framework

```
User query
   ↓
Thought  → what Pandas operation would answer this?
Action   → df.shape[0]
Observation → 768
Thought  → that's the final answer
Final Answer → "There are 768 rows."
```

## Tech stack

![LangChain](https://img.shields.io/badge/LangChain-0.2-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green)
![Pandas](https://img.shields.io/badge/Pandas-2.0-orange)
