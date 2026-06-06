"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : **I ask the agent "Who is the current president of the United States?"**
Index    : 44
"""

# ---
# ### **I ask the agent "Who is the current president of the United States?"**
# ---

question = 'Who is the current president of the United States?'
user_query = {"input": question }
result =RAG_executor(user_query)
print(f"Result: {result['output']}")
