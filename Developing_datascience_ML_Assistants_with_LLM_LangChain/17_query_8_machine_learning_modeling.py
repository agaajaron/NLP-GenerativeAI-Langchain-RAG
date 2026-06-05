"""
Notebook : Developing_datascience_ML_Assistants_with_LLM_LangChain
Section  : **Query 8: Machine Learning Modeling**
Index    : 17
"""

# ---
# ## **Query 8: Machine Learning Modeling**
# ---

# ---
# Finally, we can also get the LLM agent to perform Machine Learning modeling. Let us attempt to create a Logistic Regression model that aims to classify whether a person is likely to have diabetes or not given their health measurements.
#
# Since Machine Learning is a multi-step process - we will need to spell out the exact steps the agent is required to follow, the model it needs to use (Logistic Regression) and the outputs we expect from it (the Confusion Matrix and the Accuracy Score) - as such, a longer, step-by-step prompt will be required.
# ---

user_message = 'Follow these steps to run a Logistic Regression model to predict the Outcome variable:\
1. Clean the dataset so it has no null values.\
2. Preprocess the dataset so it is ready for use by a Logistic Regression model.\
3. Run Logistic Regression to classify people into Outcome 1 or Outcome 0 with the following steps:\
3a. Split the data into Train and Test sets. These datasets should not contain the Outcome column.\
3b. Train the model on the Train dataset. Make sure the model converges, so use as many iterations as necessary for that.\
3c. Obtain y_pred by predicting on X_test and use a threshold to convert y_pred into binary values.\
3d. Plot the Confusion Matrix, print the accuracy score between binary y_pred and y_test, and comment on the predictive performance of the model.'

agent.run(user_message);


if __name__ == "__main__":
    pass  # entry point
