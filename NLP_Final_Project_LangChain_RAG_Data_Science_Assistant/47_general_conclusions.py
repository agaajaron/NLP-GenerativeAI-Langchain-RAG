"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : General conclusions:
Index    : 47
"""

# ---
# # General conclusions:
#
# 1)Langchain allows for designing assistant agent that works exetremely fast and very intuitive.
#
# 2)The design of the Langchaine agent process is consistent and one can follow steps of the "Think Act Observe" steps performed. Depending on the result of the run of the agent with given prompt one can modify the prompt to specify in more detail what is expected and computed.  
#
# 3)Pandas agent offers very fast way of analyzing dataframes. It can be useful in many instances for exmaple when one has unfamiliar dataset and wishes to learn quickly specific information or when one compares datasets, for example  when one has large - similar or not so similar dataframes. In particular when one compares data from slightly different dataframes langchain data science assistant could help to find information very quickly. I can imagine that also the mistakes the langchain assitant would make (if it makes mistakes) would be different than human mistakes which can help avoid mistakes.
#
# 4)Comparing distributions or plotting heatmap is very fast and simple with the assistant.
#
# 5) Some of the requests about the trends in the data did not result in acceptable answer. But trying to run the same prompt couple of times would generate required output sooner or later. It seems a bit random. Sometimes the first run is correct and the answers are as expected. I comment more on the issue below.
#
# 6) Machine Learning models for simple models of binary classification prompts worked very well. Couple of times I have seen strange output (see comments below) but most of the times agent performed what was described in the prompt and gave the consistent output that seems correct.
# The results were obtained fast and seem reasonable.
#
# 7) When given selection of machine learning models to choose for classification the agent chose random forest model and provided reasonable explanation: "I chose Random Forest because it is a versatile algorithm that can handle both numerical and categorical data, it performs well with large datasets, and it also reduces overfitting by averaging the result of different decision trees.".
#
#
# 8) Running machine learning experiments might require more consistency so that each run is of the same quality. That might be helped with more detailed prompt. I will try this to see how the "randomness" in answering changes, if one can get more "deterministic"  behavior.
# In a work environment one can add a layer with iterative runs of the agent until condition for checking of the answer is fullfilled, for example to avoid certain "phrases" in the outputs. One can have a simple function checking if answer of the agent contains certain words (and then 'repeat the run' or 'accept') or run the retrieval agent on the answer of the pandas agent and do some "quality check" in this way.  
#
# 9) Machine Learning neutral network prompt requesting Tensor Flow code - worked well. Very fast and simple. Within seconds - one gets working model with a rather simple prompt. However for simple architecture of 4 hidden layers I had already too many tokens, so I had to reduce number of layers which made the accuracy similar to previous models i.e. 0.7. But one can certainly with more tokens improve accuracy.
#
# 10) As an extra - I tried also pytorch version. But machine learning netural network prompt requesting pytorch code model did not produce satisfactory outcome. The agent was not providing the correct code.For example I requested hidden layers and specific loss function but it did not use ReLu after hidden layers or there were other bugs in the code - specific for the pytorch syntax (see below). Specifying the NN architecture in even more detail in the prompt could take longer than using/modifying existing pytorch code for such a simple neural network. I have tried simple prompt and then also using lines of code sample in the prompt but the result was still wrong (it would have to be debugged and run separately without langchain - at least in present state). The result of this attemtp is copied below.
#
# 11) All machine learning models were of similar accuracy 0.69-0.7.
#
# 12) SQL agent worked very well and fast. I did not find any issues with output.
# Exremely easy tool.Each run was without suprises. It was very simple with nicely formulated answers and correct code. One get information without exact memorization of the syntax.
#
# 13) Retrieval Augmented Generation agent worked very well and fast. All questions were answered correctly with more than satisfactory quality of longer answers.
#
# 14) I think that using all of the different tools/agents and data types and data sources in one notebook is also very impressive. One can connect different sources of data - datasets from csv files or databases and specific text documents and work on a project combining all of information into one coherent work flow.  
#
# 15) I think that the langchain assistant can make great 'impression' and
# when used by qualified data scientist it can be very useful tool. It can be used as simple -no coding- fast tool for preliminary inquiries but also for more detailed inquiries of data from different sources and quickly manipulating the data starting fromm the scratch with relatively short prompt that generates infomration, text or code that can be used further or can be immediately visualized.
#
# 16) The only concern is reproducibility - and 'determinism'. It can help to
# run the agent couple of times and save /freeze the 'best' outcome. I did not have to run it more than 6 times for one prompt to see the answers/results start to repeat.
#
# 17) The quality of machine learning models were limited by number of tokens and this can be further studied.
#
# 18) It was very interesting project that shows that in order to efficiently write useful prompt for langchain datascience assistant one needs to understand data science topics, machine learning topics, langchain agents and basics of language models and the natural language learning specialization (question answering, information retrieval etc.).   
#
#
#
#
#
#
# ### Attempt with Pytorch:
# ---

nn_model_prompt2 = 'Run neural network binary classification model coded in Pytorch. Use architecture: self.hidden = nn.Linear(number of features, 2* number of features), then self.relu = nn.ReLU(), then self.output = nn.Linear(2*number of features, 1), self.sigmoid = nn.Sigmoid()\
Follow the steps below to run the model to predict the Outcome variable:\
1. Clean the dataset so it has no null values.\
2. Preprocess the dataset so it is ready for use by a Binary Classification model.\
3. Run Binary Classification to classify people into Outcome 1 or Outcome 0 with the following steps:\
3a. Split the data into Train and Test sets. These datasets should not contain the Outcome column.\
3b. Train the model on the Train dataset. Make sure the model converges, so use as many iterations as necessary for that.\
3c. Obtain y_pred by predicting on X_test and use a threshold to convert y_pred into binary values.\
3d. Plot the Confusion Matrix, print the accuracy score between binary y_pred and y_test, and comment on the predictive performance of the model.';
agent.run(nn_model_prompt2)


if __name__ == "__main__":
    pass  # entry point
