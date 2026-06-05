"""
Notebook : NLP_Final_Project_LangChain_RAG_Data_Science_Assistant
Section  : **XIV. Retrieval-Augmented Generation**
Index    : 42
"""

# ---
# ## **XIV. Retrieval-Augmented Generation**
# ---

# ---
# **In this section, I perform Retrieval-Augmented Generation (RAG) with a LangChain agent.**
#
# RAG refers to the process of supplementing the context of an LLM from the System Prompt and User Query, with additional chunks of text that may have been retrieved from external sources such as a Document or the results of a Web Search. This additional context is used by the LLM to generate what usually turns out to be a much more contextual and relevant answer to the User Query, especially when that query refers to current news or specific facts or figures that may not have been present in the LLM's original pre-training dataset. **The benefits of augmenting an LLM's capability with RAG are numerous**, such as providing it with additional knowledge, reducing its risk of hallucination, and being able to now fact-check the LLM by monitoring the sources from which it has generated its answer.
#
# - What I do here is first load our document, split it into easy to handle chunks, and then vectorize and store these chunks in a Vector Database
# - Then provide our RAG Agent with this Vector Database and a few more tools
# - Whenever we ask the agent a question, it will refer to the document to find something from the database which could answer our question
#
# ---

from langchain.agents.agent_toolkits import create_retriever_tool

# ---
# I shall import a chapter from an Operating Systems course book.
# ---

# Let's create an object to load the book
Loader = PyPDFLoader('https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-intro.pdf')

# ---
# Next, we shall load the document and split it into several chunks, each of size 1000 and with an overlap of 50 between chunks.
# ---

# We use the loader created above to load the document
documents = Loader.load()

# We split the document into several chunks as mentioned above
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# ---
# Let's vectorize these chunks into vector embeddings, and store them in the FAISS Vector Database.
# ---

# Now we transform these chunks into the list of number called vectors and store them into our database
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# ---
# Now, create a Retriever instance that will perform similarity search and extract those vector chunks relevant to the user query.
# ---

# We create a retreiver instance. Its job is to fetch for us a vector (portion of the document) relevant to the user
# question
retriever = db.as_retriever()

# ---
# Now I write a prompt to create the RAG agent.
# ---

# This is the prompt to create a RAG agent for us
retriever_name = "search_the_text_of_pdf"
retriever_desc = """The purpose of this tool is to answer questions based on the book: 'Operating Systems 3 \
easy pieces' regarding operating systems in computer science. Answer any query input by the user from the vectorbase\
if you do not know the answer, then use the serpapi tool to search the web for the answer. Clearly state that you\
searched the web. Keep your answers short and precise."""

# ---
# Two tools below:
#
# 1. **RAG Tool:** This is the tool that performs RAG for us on the loaded document. It needs access to the retreiver object that we created above to fetch relevant pieces from our database to answer our questions.
#
# 2. **Serpapi Tool:** In case the LLM is unable to find an answer for some question from the document that we uploaded, one option is to tell you that it did not find any relevant answer from the document, so it would need to search the internet to give you an answer. The Serpapi tool searches Google for the result.
# ---

rag_tool = create_retriever_tool(
    retriever,
    retriever_name,
    retriever_desc
)

os.environ['SERPAPI_API_KEY'] = '4bde37c74068633153b825cb3ff392ee3a6697a630674d5abb5d55be50f58a49'
search_tool = load_tools(['serpapi'])
tools = [rag_tool, search_tool[0]] # we can have multiple tools, hence the list

RAG_executor = create_conversational_retrieval_agent(llm=llm, tools=tools, verbose=True) # setting verbose=True to output the thought process of the agent

# ---
# Now, as an example, one can ask the RAG agent a question relevant to the book: "What is a process?"
#
# The model correctly calls the search_document tool to fetch us the answer to our question.
# ---

question = "what is a process"
user_query = {"input": question}
result = RAG_executor(user_query)
print(f"Result: {result['output']}")
