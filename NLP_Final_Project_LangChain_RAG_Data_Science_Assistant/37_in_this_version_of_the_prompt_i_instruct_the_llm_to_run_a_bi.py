"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : **In this version of the prompt, I instruct the LLM to run a binary classification Neural Network using TensorFlow or PyTorch to predict whether people have CHD or not. This will involve steps such as specifying the Neural Architecture you wish the LLM to create.**
Index    : 37
"""

# ---
# ### **In this version of the prompt, I instruct the LLM to run a binary classification Neural Network using TensorFlow or PyTorch to predict whether people have CHD or not. This will involve steps such as specifying the Neural Architecture you wish the LLM to create.**
# ---

# Remember to be as precise in your instructions as possible. This just helps to reduce the probability of the
# model mis-interpreting our commands

nn_model_prompt = 'Run neural network binary classification model coded in TensorFlow. Use 2 hidden layers and binary cross entropy as loss function. Follow the steps below to run the model to predict the Outcome variable:\
1. Clean the dataset so it has no null values.\
2. Preprocess the dataset so it is ready for use by a Binary Classification model.\
3. Run Binary Classification to classify people into Outcome 1 or Outcome 0 with the following steps:\
3a. Split the data into Train and Test sets. These datasets should not contain the Outcome column.\
3b. Train the model on the Train dataset. Make sure the model converges, so use as many iterations as necessary for that.\
3c. Obtain y_pred by predicting on X_test and use a threshold to convert y_pred into binary values.\
3d. Plot the Confusion Matrix, print the accuracy score between binary y_pred and y_test, and comment on the predictive performance of the model.';
agent.run(nn_model_prompt)
