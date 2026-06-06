"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : **Now  a prompt to ask the Database Agent what was the maximum salary paid to an employee, and to whom it was paid**
Index    : 40
"""

# ---
# ### **Now  a prompt to ask the Database Agent what was the maximum salary paid to an employee, and to whom it was paid**
# ---

# Retrieving the maximum salary and who received it
prompt = 'What was the maximum salary paid to an employee. What is the name of the employee who recieved the highest salary.';
db_agent.run(prompt)

# ---
# Let's now give the agent a chance to use its math tool with an arithmetic question.
# ---
