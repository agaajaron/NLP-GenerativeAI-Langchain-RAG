"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : Initialize a DataFrame Agent of the Pandas library in LangChain
Index    : 13
"""

# ---
# ###  Initialize a DataFrame Agent of the Pandas library in LangChain
# ---

# Initializing the DataFrame agent of the Pandas library in LangChain allows our LLM to work with
# Pandas DataFrames with simple, natural language commands
agent = create_pandas_dataframe_agent(llm = llm, df = df, verbose = True);
