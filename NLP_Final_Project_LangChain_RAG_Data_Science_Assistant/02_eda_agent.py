"""
LangChain RAG Data Science Assistant — EDA Agent
Loads the South Africa Heart Disease dataset and runs EDA queries
(column info, missing values, stats, KDE plots, heatmap).

Dataset: https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data
"""

import requests
import pandas as pd
from setup import make_pandas_agent


def load_dataset(path: str = "sahd.csv") -> pd.DataFrame:
    url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return pd.read_csv(path)


EDA_QUERIES = {
    "row_count":    "How many rows and columns are in the dataset?",
    "missing":      "Are there any missing values? If so, which columns?",
    "avg_obesity":  "What is the average obesity value?",
    "summary":      "Give a brief statistical summary of all numeric columns.",
    "kde_plot": (
        "In one figure, plot a KDE of obesity for people WITH CHD and "
        "WITHOUT CHD. Label both axes, add a legend, and comment on the "
        "difference in distributions."
    ),
    "age_dist": (
        "Plot the distribution of age with respect to CHD status. "
        "Label the axes and comment on the main trend."
    ),
    "heatmap": (
        "Create a pairwise correlation heatmap for all numeric columns. "
        "Label the axes and highlight the top-3 strongest correlations."
    ),
}

if __name__ == "__main__":
    df = load_dataset()
    agent = make_pandas_agent(df)

    for name, query in EDA_QUERIES.items():
        print(f"\n{'='*60}\n[{name}] {query}\n{'='*60}")
        agent.run(query)
