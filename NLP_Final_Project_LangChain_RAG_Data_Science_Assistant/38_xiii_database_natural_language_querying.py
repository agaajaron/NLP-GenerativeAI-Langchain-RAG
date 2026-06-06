"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : **XIII. Database Natural Language Querying**
Index    : 38
"""

# ---
# ## **XIII. Database Natural Language Querying**
# ---

# ---
# I will focus on creating a new agent with our LLM.
#
# This agent will carry out 2 main tasks for us:
#
# - **Database Querying:** Think of an e-commerce website such as Amazon - they would be required to deal with a humongous number of products and customers on a daily basis. They likely store all this information in a large database - and being able to query and retrieve relevant information from that database is critical for their operations to work. This is done through queries and requests, SQL queries in particular for relational databases. We shall now ask our agent to do this for us on a toy dataset.
#
# - **Math:** Secondly, I will also equip our agent with another tool - math.
# ---

# ---
# First, we'll import the sub-packages that I will need:
# ---

from langchain.agents import *
from langchain.sql_database import SQLDatabase
from langchain.llms import HuggingFacePipeline

# ---
# In the following code block, we will be creating a toy SQL database using SQLite for demonstration purposes.
# ---

# Creating a dummy SQL database for our example.
conn = sqlite3.connect('sample.db')

# Creating a cursor object to interact with the database
cursor = conn.cursor()

# Creating a table to store employee information
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Employees (
        EmployeeID INTEGER PRIMARY KEY,
        FirstName TEXT,
        LastName TEXT,
        Birthdate DATE,
        Department TEXT,
        Salary REAL
    )
''')

# Inserting some sample data into the Employees table
cursor.executemany('''
    INSERT INTO Employees (FirstName, LastName, Birthdate, Department, Salary)
    VALUES (?, ?, ?, ?, ?)
''', [
    ('John', 'Doe', '1990-05-15', 'HR', 50000.00),
    ('Jane', 'Smith', '1985-12-10', 'Sales', 55000.00),
    ('Bob', 'Johnson', '1992-08-25', 'Engineering', 60000.00),
    ('Alice', 'Brown', '1988-04-03', 'Marketing', 52000.00)
])

# Commiting the changes and close the connection
conn.commit()
conn.close()

# ---
# In place of the above code, one can also enter the path to our database file directly.
# ---

# In place of file, you can enter the path to your database file
file = "sample.db"
db = SQLDatabase.from_uri(f"sqlite:///{file}")

# ---
# Now I create our Database Agent and equip it with the LLM math tool.
# ---

tools = ['llm-math']

tools = load_tools(tools,llm)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
db_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    tools=tools,
    verbose=True
)

# ---
# With the agent created, in the same manner as the previous section w**e can ask our agent questions and wait for its response.**
#
# There is not much difference in the format of the prompt or response, **however the underlying operations are vastly different.**
# ---
