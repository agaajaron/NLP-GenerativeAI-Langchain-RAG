"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : Add the name of the GPT-4 model that I use as the LLM for the LangChain Assistant
Index    : 09
"""

# ---
# ### Add the name of the GPT-4 model that I use as the LLM for the LangChain Assistant
# ---

# This is our LLM, the heart of the entire process. Here we are using the GPT-4 Model from OpenAi.
llm = ChatOpenAI(model_name="gpt-4")
