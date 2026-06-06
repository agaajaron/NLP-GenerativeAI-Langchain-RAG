"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : **I will ask the agent "What are the different states of a process?"**
Index    : 43
"""

# ---
# ### **I will ask the agent "What are the different states of a process?"**
# ---

question = 'What are the different states of a process?'
user_query = {"input": question }
result =RAG_executor(user_query)
print(f"Result: {result['output']}")

# ---
# Let's now ask the other kind of question, where the answer to the question may not be present in the PDF document, so the agent has to recognize that and turn to the Search tool to search the web and find a possible answer from there.
# ---
