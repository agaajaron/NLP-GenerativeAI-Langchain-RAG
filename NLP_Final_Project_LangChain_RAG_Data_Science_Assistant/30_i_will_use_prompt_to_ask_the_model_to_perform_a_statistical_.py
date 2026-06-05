"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : **I will use prompt to ask the model to perform a statistical t-Test to validate the following hypotheses:**
Index    : 30
"""

# ---
# ### **I will use prompt to ask the model to perform a statistical t-Test to validate the following hypotheses:**
#
# ### **Null Hypothesis:** Higher levels of obesity do not increase the risk of CHD.
#
# ### **Alternate Hypothesis:** Higher levels of obesity increase the risk of CHD. **(5 marks)**
# ---

# Let's use a t-value Hypothesis Test to check whether obesity affects the chances of CHD
text = 'Validate the following hypothesis using a T-test. Null Hypothesis: Higher levels of obesity do not increase the risk of CHD.Alternate Hypothesis: Higher levels of obesity increase the risk of CHD.'
agent.run(text)
