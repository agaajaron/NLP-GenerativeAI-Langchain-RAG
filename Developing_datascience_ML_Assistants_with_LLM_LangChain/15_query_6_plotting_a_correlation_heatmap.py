"""
Notebook : Developing_datascience_ML_Assistants_with_LLM_LangChain
Section  : **Query 6: Plotting a Correlation Heatmap**
Index    : 15
"""

# ---
# ## **Query 6: Plotting a Correlation Heatmap**
# ---

# ---
# In this example, we shall ask the agent to plot a correlation heatmap among the columns and comment on the trends observed in terms of which columns seem most strongly correlated with each other. We shall see that the LLM is capable of servicing this more complex request.
# ---

user_message = 'Plot a correlation heatmap, showing correlations among all the columns. Label the axes, and comment on the main trends.';
agent.run(user_message);

# ---
# As we can see from the answer above, the agent correctly surmises a couple of more meaningful pairwise correlations - the Glucose/Outcome pair and the Age/Pregnancies pair. **In addition to just pointing out that these correlations are relatively strong, the agent also attempts to give a trend explanation** by saying for example, that "a higher glucose level could indicate a higher likelihood of diabetes" or "as a woman ages, she is likely to have had more pregnancies" - showing that the LLM has Natural Language Understanding of the the meanings of the column names.
# ---
