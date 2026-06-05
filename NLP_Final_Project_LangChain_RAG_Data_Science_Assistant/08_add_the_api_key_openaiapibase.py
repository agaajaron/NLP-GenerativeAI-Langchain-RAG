"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : Add the API Key & OPENAI_API_BASE
Index    : 08
"""

# ---
# ### Add the API Key & OPENAI_API_BASE
# ---

# Importing the GPT-4 LLM and setting it up
API_KEY = "gl-U2FsdGVkX1+E2fVRSm1LmnTdMR2KaOTfDzOXrkapr/bPxZkxtfkX3tXE27XsKOlY"
OPENAI_API_BASE = "https://aibe.mygreatlearning.com/openai/v1"

os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE
