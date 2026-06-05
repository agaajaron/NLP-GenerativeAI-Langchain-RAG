"""
Notebook : Developing_datascience_ML_Assistants_with_LLM_LangChain
Section  : **Query 7: Hypothesis Testing**
Index    : 16
"""

# ---
# ## **Query 7: Hypothesis Testing**
# ---

# ---
# In addition to the EDA style queries we have been asking the LLM to service above, it is also capable of performing tasks like Hypothesis Testing to statistically validate a quantitative claim we may have about the data - a useful endeavor when it comes to Data Science decision making. In the following example, we shall ask the LLM to use a T-test to validate whether higher levels of glucose increase the likelihood of diabetes or not.
# ---

user_message = 'Validate the following hypothesis using a T-test. Null Hypothesis: Higher levels of Glucose do not increase the risk of Outcome 1. Alternate Hypothesis: Higher levels of Glucose increase the risk of Outcome 1.';
agent.run(user_message);
