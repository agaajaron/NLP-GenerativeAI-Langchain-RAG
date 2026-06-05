"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : **Example of a prompt to ask the Database Agent to compute the square root of the maximum salary paid to an employee**
Index    : 41
"""

# ---
# ### **Example of a prompt to ask the Database Agent to compute the square root of the maximum salary paid to an employee**
# ---

# Let's see if our LLM is profecient at arithemtic with the help of its math tool
prompt = 'Compute the square root of the maximum salary paid to an employee';
db_agent.run(prompt)
