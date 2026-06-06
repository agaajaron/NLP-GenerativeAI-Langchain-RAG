"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : I will instruct the model to display a heatmap to show the correlation between all the columns.
Index    : 27
"""

# ---
# ###  I will instruct the model to display a heatmap to show the correlation between all the columns.
# ---

text = 'Create pairwise correlation heatmap to show correlation between all of the columns. Label the axes. Comment on the trends for example which columns are correlated.';
agent.run(text)

# ---
# Alright, so as we see, our model is quite good at plotting requests and finding the underlying trends.
#
# Let's see if it can go to the next level and also test out hypothesese for us.
# ---
