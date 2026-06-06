"""
Notebook : Developing_datascience_ML_Assistants_with_LLM_LangChain
Section  : **Creating the Pandas DataFrame Agent**
Index    : 09
"""

# ---
# ## **Creating the Pandas DataFrame Agent**
# ---

# ---
# The particular type of LangChain agent we shall be working with is a called a Pandas DataFrame agent - this agent is a specific instance of using an LLM such as GPT-4 to analyze a Pandas DataFrame with custom prompts and instructions provided by LangChain to the LLM for that purpose. We shall also set the verbose parameter in the following function to True, so that we can follow the internal thought process behind how the LLM arrives at its answers.
# ---

agent = create_pandas_dataframe_agent(llm = llm, df = df, verbose = True);
