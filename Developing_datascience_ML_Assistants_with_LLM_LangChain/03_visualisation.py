"""
LangChain Data Science Assistant — Project 1
Visualisation: correlation heatmap and hypothesis testing via LLM agent.
"""

import pandas as pd
from setup import make_agent

df = pd.read_csv("diabetes.csv")
agent = make_agent(df)

PLOT_QUERY = (
    "Plot a correlation heatmap showing correlations among all columns. "
    "Label both axes and comment on the two strongest correlations you observe."
)

HYPOTHESIS_QUERY = (
    "Use a two-sample t-test to validate the following hypothesis:\n"
    "  Null: Higher Glucose levels do NOT increase the risk of Outcome 1.\n"
    "  Alt:  Higher Glucose levels DO increase the risk of Outcome 1.\n"
    "Report the t-statistic, p-value, and your conclusion."
)

if __name__ == "__main__":
    print("=== Correlation Heatmap ===")
    agent.run(PLOT_QUERY)

    print("\n=== Hypothesis Test ===")
    agent.run(HYPOTHESIS_QUERY)
