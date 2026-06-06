"""
LangChain RAG Data Science Assistant — RAG Pipeline
Loads a PDF, chunks and embeds it into FAISS, then creates a conversational
retrieval agent that answers from the document or falls back to web search.
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import load_tools
from langchain.agents.agent_toolkits import (
    create_conversational_retrieval_agent,
    create_retriever_tool,
)
from setup import llm

PDF_URL = "https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-intro.pdf"

TOOL_NAME = "search_os_textbook"
TOOL_DESC = (
    "Search 'Operating Systems: Three Easy Pieces' for questions about "
    "processes, scheduling, memory, and concurrency. If the answer isn't in "
    "the book, use the web search tool instead."
)

RAG_QUESTIONS = [
    "What are the different states of a process?",
    "Who is the current president of the United States?",   # out-of-scope → web
    "Who is the CEO of Microsoft?",                          # out-of-scope → web
]


def build_rag_agent():
    # 1. Load and split the PDF
    docs   = PyPDFLoader(PDF_URL).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50
    ).split_documents(docs)

    # 2. Embed and store in FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 3. Create a retriever tool + optional SerpAPI fallback
    rag_tool = create_retriever_tool(retriever, TOOL_NAME, TOOL_DESC)
    tools    = [rag_tool]

    if os.getenv("SERPAPI_API_KEY"):
        web_tool = load_tools(["serpapi"])[0]
        tools.append(web_tool)

    return create_conversational_retrieval_agent(llm=llm, tools=tools, verbose=True)


if __name__ == "__main__":
    agent = build_rag_agent()

    for question in RAG_QUESTIONS:
        print(f"\n{'='*60}\nQ: {question}\n{'='*60}")
        result = agent({"input": question})
        print("A:", result["output"])
