"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : **Here is  a prompt to ask the Database Agent to name the columns present in the database, and then run that prompt with the `db_agent` defined above**
Index    : 39
"""

# ---
# ### **Here is  a prompt to ask the Database Agent to name the columns present in the database, and then run that prompt with the `db_agent` defined above**
# ---

# Naming the columns present in the database
prompt = 'Name the columns present in the database';
db_agent.run(prompt);
