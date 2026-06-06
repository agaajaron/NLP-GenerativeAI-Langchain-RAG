# NLP_Final_Project_LangChain_RAG_Data_Science_Assistant

Perhaps the largest and most prominent application of Large Language Models to date has been their ability to converse w

## Files

| File | Section | Lines |
|------|---------|-------|
| `06_importing_the_libraries_into_the_notebook.py` | **Importing the libraries into the notebook** | 35 |
| `08_add_the_api_key_openaiapibase.py` | Add the API Key & OPENAI_API_BASE | 17 |
| `09_add_the_name_of_the_gpt_4_model_that_i_use_as_the_llm_for_th.py` | Add the name of the GPT-4 model that I use as the LLM for the LangChain Assistant | 13 |
| `12_use_the_readcsv_function_to_import_the_south_africa_heart_di.py` | Use the read_csv() function to import the South Africa Heart Disease CSV dataset provided | 22 |
| `13_initialize_a_dataframe_agent_of_the_pandas_library_in_langch.py` | Initialize a DataFrame Agent of the Pandas library in LangChain | 14 |
| `14_i_write_a_user_query_prompt_to_ask_the_agent_how_many_rows_t.py` | I write a user query prompt to ask the agent how many rows the DataFrame contains, and run the agent command | 28 |
| `16_now_i_find_out_the_names_of_the_different_columns_present_in.py` | Now I find out the names of the different columns present in this dataset | 13 |
| `17_i_find_out_how_many_missing_values_are_present_in_the_datase.py` | I find out how many missing values are present in the dataset | 13 |
| `18_one_can_compute_the_average_obesity_value.py` | One can compute the average obesity value | 16 |
| `20_i_get_a_brief_summary_of_the_columns_present_in_the_dataset.py` | I get a brief summary of the columns present in the dataset | 17 |
| `23_i_instruct_the_model_to_generate_a_kde_plot_of_obesity_among.py` | I instruct the model to generate a KDE plot of obesity among people with / without CHD. I will ask for axes lebels. | 16 |
| `25_i_will_prompt_the_model_to_show_the_distribution_of_age_of_p.py` | I will prompt the model to show the distribution of age of people with respect to CHD. (I want the label for the axes, and comments on the principal trend observed). | 17 |
| `27_i_will_instruct_the_model_to_display_a_heatmap_to_show_the_c.py` | I will instruct the model to display a heatmap to show the correlation between all the columns. | 19 |
| `30_i_will_use_prompt_to_ask_the_model_to_perform_a_statistical_.py` | **I will use prompt to ask the model to perform a statistical t-Test to validate the following hypotheses:** | 18 |
| `33_i_write_now_a_detailed_step_by_step_prompt_requesting_the_mo.py` | **I write now a detailed step-by-step prompt requesting the model to create a Logistic Regression model using the steps common to the ML workflow, such as cleaning the dataset, preprocessing it, splitting the data into Train and Test, training the model on the Train dataset and making predictions on the Test dataset. The model should print out the accuracy score it obtains on the Test dataset.** | 25 |
| `35_i_write_now_a_similar_detailed_step_by_step_prompt_to_the_ab.py` | **I write now a similar detailed step-by-step prompt to the above, this time requesting the LLM to run a Binary Classification model of its choice from a list of options we give it, such as Decision Trees, Random Forests, XGBoost, Support Vector Machines, Neural Network, etc. The LLM has to provide reasoning for why it picked this model. Otherwise, the rest of the steps can be similar to the prompt drafted for the earlier question.** | 24 |
| `37_in_this_version_of_the_prompt_i_instruct_the_llm_to_run_a_bi.py` | **In this version of the prompt, I instruct the LLM to run a binary classification Neural Network using TensorFlow or PyTorch to predict whether people have CHD or not. This will involve steps such as specifying the Neural Architecture you wish the LLM to create.** | 23 |
| `38_xiii_database_natural_language_querying.py` | **XIII. Database Natural Language Querying** | 95 |
| `39_here_is_a_prompt_to_ask_the_database_agent_to_name_the_colum.py` | **Here is  a prompt to ask the Database Agent to name the columns present in the database, and then run that prompt with the `db_agent` defined above** | 14 |
| `40_now_a_prompt_to_ask_the_database_agent_what_was_the_maximum_.py` | **Now  a prompt to ask the Database Agent what was the maximum salary paid to an employee, and to whom it was paid** | 18 |
| `41_example_of_a_prompt_to_ask_the_database_agent_to_compute_the.py` | **Example of a prompt to ask the Database Agent to compute the square root of the maximum salary paid to an employee** | 14 |
| `42_xiv_retrieval_augmented_generation.py` | **XIV. Retrieval-Augmented Generation** | 99 |
| `43_i_will_ask_the_agent_what_are_the_different_states_of_a_proc.py` | **I will ask the agent "What are the different states of a process?"** | 19 |
| `44_i_ask_the_agent_who_is_the_current_president_of_the_united_s.py` | **I ask the agent "Who is the current president of the United States?"** | 15 |
| `45_i_ask_the_agent_to_find_out_who_is_the_ceo_of_microsoft.py` | **I ask the agent to find out "Who is the CEO of Microsoft?"** | 21 |
| `47_general_conclusions.py` | General conclusions: | 70 |

## Requirements

```
cohere
faiss-gpu
google-search-results
langchain
langchain_experimental
matplotlib
openai
pandas
pypdf
requests
seaborn
tiktoken
```
