"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : **I write now a similar detailed step-by-step prompt to the above, this time requesting the LLM to run a Binary Classification model of its choice from a list of options we give it, such as Decision Trees, Random Forests, XGBoost, Support Vector Machines, Neural Network, etc. The LLM has to provide reasoning for why it picked this model. Otherwise, the rest of the steps can be similar to the prompt drafted for the earlier question.**
Index    : 35
"""

# ---
# ### **I write now a similar detailed step-by-step prompt to the above, this time requesting the LLM to run a Binary Classification model of its choice from a list of options we give it, such as Decision Trees, Random Forests, XGBoost, Support Vector Machines, Neural Network, etc. The LLM has to provide reasoning for why it picked this model. Otherwise, the rest of the steps can be similar to the prompt drafted for the earlier question.**
# ---

variable_model_prompt = 'Run Binary Classification model, using one of the following models: Decision Tree, Random Forest, XGBoost, Support Vector Machines, Neural Network. Provide reason for particular model selection. Follow these steps to run the model to predict the Outcome variable:\
1. Clean the dataset so it has no null values.\
2. Preprocess the dataset so it is ready for use by a Binary Classification model.\
3. Run Binary Classification to classify people into Outcome 1 or Outcome 0 with the following steps:\
3a. Split the data into Train and Test sets. These datasets should not contain the Outcome column.\
3b. Train the model on the Train dataset. Make sure the model converges, so use as many iterations as necessary for that.\
3c. Obtain y_pred by predicting on X_test and use a threshold to convert y_pred into binary values.\
3d. Plot the Confusion Matrix, print the accuracy score between binary y_pred and y_test, and comment on the predictive performance of the model.';
agent.run(variable_model_prompt)

# ---
# Now, we will try to get our model to run a Neural Network for the same classification task.
# ---
