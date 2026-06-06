"""
LangChain Data Science Assistant — Project 1
EDA Queries: row counts, column names, missing values, averages, summary.

Dataset: Pima Indians Diabetes (diabetes.csv)
"""

import pandas as pd
from setup import make_agent

df = pd.read_csv("diabetes.csv")
agent = make_agent(df)

EDA_QUERIES = [
    "How many rows are there?",
    "How many columns are there, and what are their names?",
    "Are there any missing values in the dataset?",
    "What is the average Age in the dataset?",
    "Give me a brief statistical summary of every column.",
]

if __name__ == "__main__":
    for q in EDA_QUERIES:
        print(f"\n{'='*60}\nQuery: {q}\n{'='*60}")
        agent.run(q)
