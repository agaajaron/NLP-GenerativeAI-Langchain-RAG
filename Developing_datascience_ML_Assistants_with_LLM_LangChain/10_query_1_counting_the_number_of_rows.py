"""
Notebook : Developing_datascience_ML_Assistants_with_LLM_LangChain
Section  : **Query 1: Counting the number of rows**
Index    : 10
"""

# ---
# ## **Query 1: Counting the number of rows**
# ---

# ---
# We can start off with a simple query - let's ask the agent for the number of rows present in this dataset.
# ---

user_message = 'How many rows are there?';
agent.run(user_message);

# ---
# As we can see from above, **the LLM agent uses a particular framework provided by LangChain (popularly called the ReAct Framework - Reason and Act)**, which forces it to carry out a structured thinking process in order to respond to our query. This is subtly different from interacting directly with an assistant-style LLM with our own query and system prompt - in this scenario, LangChain already has a system prompt in place for its agents that asks the LLM to use the ReAct Framework (with Thoughts, Actions and Observations) to come up with the Final Answer.  
#
# In the example above, the LLM first generates a Thought on how it can find the number of rows in the DataFrame. This is then converted to an Action, the Pandas code that corresponds to the Thought, and that then leads to an Observation (which is the output of that Pandas code). The LLM then again reasons with itself using another Thought, which deduces that this Observation is in fact the final answer it is required to give, and it frames a Final Answer based on that finding.
#
# The number of these Thought - Action - Observation cycles will vary depending on the complexity of the user query - the more complex the query, the higher the number of such cycles required by the LLM agent to give us our final answer, which in a way, mirrors how we humans arrive at answers to questions of varying complexity with our reasoning process as well.
#
# In the following examples, let's ask EDA-style questions of increasing complexity to the LLM.
# ---
