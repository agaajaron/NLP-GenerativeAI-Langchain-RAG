"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : I will prompt the model to show the distribution of age of people with respect to CHD. (I want the label for the axes, and comments on the principal trend observed).
Index    : 25
"""

# ---
# ### I will prompt the model to show the distribution of age of people with respect to CHD. (I want the label for the axes, and comments on the principal trend observed).
# ---

text = 'Plot the distribution of age of people with respect to CHD. Labels the axes.Comment on the principal trend observed in the plot.';
agent.run(text)

# ---
# Let us now try to see how the various factors contributing towards heart dieases are related to each other. We can do this by plotting a heatmap where the darker colored tiles would mean that the factors corresponding to the row and column position of that tile are more heavily correlated, and lighter colors would show a lighter correlation.
# ---
