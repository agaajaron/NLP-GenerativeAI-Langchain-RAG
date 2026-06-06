"""
LangChain Data Science Assistant — Project 1
ML Modeling: prompt the agent to train, evaluate and explain a classifier.

The agent writes and executes all Pandas / Scikit-learn code autonomously;
we only supply the high-level instructions.
"""

import pandas as pd
from setup import make_agent

df = pd.read_csv("diabetes.csv")
agent = make_agent(df)

ML_PROMPT = """
Follow these steps to build a Logistic Regression classifier that predicts
the Outcome column:

1. Drop rows with null values.
2. Separate features (X) from the target (y = Outcome).
3. Standardise X with StandardScaler.
4. Split into 80/20 train-test sets (random_state=42).
5. Fit LogisticRegression(max_iter=1000).
6. Print the test accuracy and a classification report.
7. Plot and display a confusion matrix heatmap.
8. Comment briefly on model performance.
"""

if __name__ == "__main__":
    print("=== Logistic Regression via LLM Agent ===")
    agent.run(ML_PROMPT)
