"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : Use the read_csv() function to import the South Africa Heart Disease CSV dataset provided
Index    : 12
"""

# ---
# ###  Use the read_csv() function to import the South Africa Heart Disease CSV dataset provided
# ---

# Importing the structured data.
# requires: requests
import requests
req=requests.get("https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data")
url_content=req.content
cvsfile=open('cahd.csv','wb')
cvsfile.write(url_content)
cvsfile.close()
df = pd.read_csv('cahd.csv')

df
