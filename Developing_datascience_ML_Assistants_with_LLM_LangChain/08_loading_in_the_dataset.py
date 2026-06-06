"""
Notebook : Developing_datascience_ML_Assistants_with_LLM_LangChain
Section  : **Loading in the dataset**
Index    : 08
"""

# ---
# ## **Loading in the dataset**
# ---

# ---
# Like we would always do for a Data Science project, we shall begin by using the read_csv() function to load in our dataset. For our trial in this notebook, we shall be working with the [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) dataset, a popular open-source Machine Learning dataset that assesses if a person of the Pima tribe origin has diabetes or not, based on a number of health factors.
# ---

df = pd.read_csv('diabetes.csv');
