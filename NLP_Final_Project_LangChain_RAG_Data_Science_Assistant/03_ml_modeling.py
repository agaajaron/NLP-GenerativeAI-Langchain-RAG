"""
LangChain RAG Data Science Assistant — ML Modeling
Hypothesis testing + three classifiers (Logistic Regression, Random Forest,
Neural Network) all driven by natural-language prompts.
"""

import pandas as pd
from eda_agent import load_dataset
from setup import make_pandas_agent

df = load_dataset()
agent = make_pandas_agent(df)

HYPOTHESIS_PROMPT = """
Use a two-sample t-test to validate:
  Null:       Higher obesity does NOT increase the risk of CHD.
  Alternate:  Higher obesity DOES increase the risk of CHD.
Report the t-statistic, p-value, and your conclusion.
"""

LOGISTIC_REGRESSION_PROMPT = """
Follow these steps to build a Logistic Regression model predicting chd:
1. Drop rows with nulls. Encode any categorical columns.
2. Separate features (X) from target (y = chd).
3. Standardise X. Split 80/20 (random_state=42).
4. Fit LogisticRegression(max_iter=1000).
5. Print accuracy and a full classification report.
6. Plot a confusion matrix heatmap.
"""

RANDOM_FOREST_PROMPT = """
Build a Random Forest classifier for chd using the same preprocessing as
the Logistic Regression above. Explain why you chose Random Forest.
Report accuracy, classification report, and feature importances (bar chart).
"""

NEURAL_NETWORK_PROMPT = """
Build a TensorFlow binary classification neural network for chd:
- Architecture: Dense(32, relu) → Dense(16, relu) → Dense(1, sigmoid)
- Loss: binary_crossentropy, Optimizer: adam, Epochs: 50
- Standardise features, 80/20 split (random_state=42)
- Print final test accuracy and plot training vs validation loss.
"""

PROMPTS = {
    "hypothesis_test":    HYPOTHESIS_PROMPT,
    "logistic_regression": LOGISTIC_REGRESSION_PROMPT,
    "random_forest":       RANDOM_FOREST_PROMPT,
    "neural_network":      NEURAL_NETWORK_PROMPT,
}

if __name__ == "__main__":
    for name, prompt in PROMPTS.items():
        print(f"\n{'='*60}\n[{name}]\n{'='*60}")
        agent.run(prompt)
