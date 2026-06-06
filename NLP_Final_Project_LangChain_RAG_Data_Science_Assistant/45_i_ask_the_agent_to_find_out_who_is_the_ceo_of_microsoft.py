"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : **I ask the agent to find out "Who is the CEO of Microsoft?"**
Index    : 45
"""

# ---
# ### **I ask the agent to find out "Who is the CEO of Microsoft?"**
# ---

question = 'Who is the CEO of Microsoft?'
user_query = {"input": question }
result =RAG_executor(user_query)
print(f"Result: {result['output']}")

# ---
# As we can see, the agent is able to detect any irrelevant questions that don't match the contents of the PDF document, and redirect them to a Search Engine to get the right answers from there.
#
# This concludes our simple demonstration of the various capabilities of LangChain Agent Assistants through this Final Project notebook.
# ---
