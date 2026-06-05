"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : I write a user query prompt to ask the agent how many rows the DataFrame contains, and run the agent command
Index    : 14
"""

# ---
# ### I write a user query prompt to ask the agent how many rows the DataFrame contains, and run the agent command
# ---

# .run enables us to pass a query into the agent and see its output
text = 'How many rows are there';
agent.run(text);

# ---
# In the code above, I store our question in the `text` variable.
#
# Then I simply `run` the agent with this question.
#
# What this does is, it forwards this question to our LLM, and lets it decide:
#
# "for this user question give me a response. Use pandas if you have to"
#
# On running this code, one should be able to see the **Chain of Thought of the LLM**, this is our model reasoning with itself to answer our question.
#
# One should be able to see that the model feels using the Pandas library (which we provided through our agent) would be a good idea!
# ---
