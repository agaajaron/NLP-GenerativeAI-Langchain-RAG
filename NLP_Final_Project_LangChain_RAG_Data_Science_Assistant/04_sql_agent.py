"""
LangChain RAG Data Science Assistant — SQL Agent
Creates an in-memory SQLite database and queries it using natural language.
"""

import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain.agents import load_tools
from setup import llm

DB_FILE = "sample.db"

EMPLOYEES = [
    ("John",  "Doe",     "1990-05-15", "Engineering", 85000.00),
    ("Jane",  "Smith",   "1985-12-10", "Sales",        72000.00),
    ("Bob",   "Johnson", "1992-08-25", "Engineering", 91000.00),
    ("Alice", "Brown",   "1988-04-03", "Marketing",   68000.00),
    ("Carol", "White",   "1995-01-20", "HR",           61000.00),
]


def create_db(path: str = DB_FILE) -> str:
    conn = sqlite3.connect(path)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS Employees")
    cur.execute("""
        CREATE TABLE Employees (
            EmployeeID  INTEGER PRIMARY KEY AUTOINCREMENT,
            FirstName   TEXT,
            LastName    TEXT,
            Birthdate   DATE,
            Department  TEXT,
            Salary      REAL
        )
    """)
    cur.executemany(
        "INSERT INTO Employees (FirstName, LastName, Birthdate, Department, Salary) "
        "VALUES (?,?,?,?,?)",
        EMPLOYEES,
    )
    conn.commit()
    conn.close()
    return path


def make_sql_agent(db_path: str = DB_FILE):
    db      = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools   = load_tools(["llm-math"], llm=llm)
    return create_sql_agent(llm=llm, toolkit=toolkit, tools=tools, verbose=True)


SQL_QUERIES = [
    "What columns are in the Employees table?",
    "Who earned the highest salary and how much was it?",
    "What is the square root of the maximum salary?",
    "Which department has the highest average salary?",
]

if __name__ == "__main__":
    db_path = create_db()
    agent   = make_sql_agent(db_path)

    for q in SQL_QUERIES:
        print(f"\n{'='*60}\nQuery: {q}\n{'='*60}")
        agent.run(q)
