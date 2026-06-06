"""
LangChain Data Science Assistant — Project 1
Setup: environment, LLM initialisation, shared utilities.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)


def make_agent(df: pd.DataFrame, verbose: bool = True):
    """Return a Pandas DataFrame agent wired to the shared LLM."""
    return create_pandas_dataframe_agent(llm=llm, df=df, verbose=verbose)


if __name__ == "__main__":
    print("LLM ready:", llm.model_name)
