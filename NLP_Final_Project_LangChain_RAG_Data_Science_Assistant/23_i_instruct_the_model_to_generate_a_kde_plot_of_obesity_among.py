"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : I instruct the model to generate a KDE plot of obesity among people with / without CHD. I will ask for axes lebels.
Index    : 23
"""

# ---
# ### I instruct the model to generate a KDE plot of obesity among people with / without CHD. I will ask for axes lebels.
#
#
# ---

# Here we will show the ability of the model to create a plot and comment on the underlying trend in the data
text = 'Generate in one figure a KDE plot of obesity among people with CHD and KDE plot of obesity among people without CHD. Label both axes and add a legend on the plot. Comment on the trend which the plot uncovers: describe properties of the distributions.';
agent.run(text)
