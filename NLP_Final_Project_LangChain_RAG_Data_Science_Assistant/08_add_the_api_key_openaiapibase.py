"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : Add the API Key & OPENAI_API_BASE
Index    : 08
"""

# ---
# ### Add the API Key & OPENAI_API_BASE
# ---

# Importing the GPT-4 LLM and setting it up
API_KEY = "apikey"
OPENAI_API_BASE = "yourapibase"

os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE
