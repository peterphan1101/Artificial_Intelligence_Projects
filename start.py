import os
import streamlit as st
import faiss
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.tools import BaseTool, Tool, tool
from langchain_openai.llms.base import OpenAI
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore

from load_companies import load_companies_to_mongo, load_companies_to_vectdb, read_sample_companies

## Credit : Coding patterns are inspired from below Oreilly's LLM course and book examples:
## Gen AI - RAG Application Development using LangChain By Manas Dasgupta
## Building LLM Powered Applications By Valentina Alto

# Load OPENAI_API_KEY, MONGO_URI and other secrets/sensitive information from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("ERROR: MongoDB connection string is not found as environment variables in .env file.")
openai_api_key = os.environ['OPENAI_API_KEY']
serpapi_api_key = os.environ['SERPAPI_API_KEY']

# Sample companies with company names and Central Index Key (CIK), a unique number assigned by the U.S. Securities and Exchange Commission (SEC)
# These companies 10-Q and 10-K data from SEC API is loaded on first question to AI bot or as part of Load Sample Data button.
# The data from these sample companies is then used to demonstrate the Retrieval Augmented Generation (RAG) functionality of the Financial Advisor application.

samples_file = "sample_companies.yml"

def query(question: str, chat_history: list):
    """
    query method uses ConversationBufferMemory and ConversationalRetrievalChain with OpenAI LLM and Vector DB retriever to generate the answers for user
    queries and save the context to memory.
    
    Parameters:
    question (str): Human/user question to AI Bot.
    chat_history (list): Chat history or context. Sequence of messages.
    
    Returns:
    Dict[str, Any]: dictionary of tuples including question, answer, chat history and documents retrieved etc.,
    """   
    memory = ConversationBufferMemory(
        return_messages=True, 
        memory_key="chat_history", 
        output_key="output"
    )
    vector_db_st = st.session_state.vector_db
    c_query = ConversationalRetrievalChain.from_llm(
        llm = st.session_state.llm,
        retriever=vector_db_st.as_retriever(), 
        return_source_documents=True)
    
    return c_query({"question": question, "chat_history": chat_history})

def initialize_load_samples():
    """
    Initializes the connection to OpenAI LLM, vector database and MongoDB clients. Loads sample data to Mongo DB, generated embeddings and stores them in FAISS
    vector database. References to Open AI LLM, Vector Database and Mongo Client are stored in session state for future retrieval of those references to use
    as part of answering the queries. Once initialized, subsequent calls do not the samples to be loaded again. As part of first time loading sample data any past data from 
    MongoDB collections is wiped out and reloaded for freshness of the data.
    
    Parameters:
    None
    
    Returns:
    None

    """
    global llm
    global vector_db
    global mongo_client

    if "is_initialized" not in st.session_state:
        llm = ChatOpenAI(temperature=0)
        embd_func = OpenAIEmbeddings()
        embd_dimension = 1536  # model = "text-embedding-ada-002"
        docstore = InMemoryDocstore()
        vector_db = FAISS(embedding_function=embd_func, index=faiss.IndexFlatL2(embd_dimension),docstore=docstore, index_to_docstore_id={}, relevance_score_fn=None, normalize_L2=False, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
        mongo_client = MongoClient(MONGO_URI)
        mongo_db = mongo_client['llm_mongo_db']
        mongo_collection = mongo_db['FinDocs_text']
        mongo_collection.drop()
        sample_companies = read_sample_companies(samples_file)
        load_companies_to_mongo(sample_companies, mongo_db)
        load_companies_to_vectdb(sample_companies, mongo_db, vector_db)
        st.session_state.vector_db = vector_db
        st.session_state.llm = llm
        st.session_state.mongo_db = mongo_db
        st.session_state.is_initialized = "True"

def main_ux():
    """
    Main user experience for human and AI bot interaction. UX is built using streamlit. Calls initialize_load_samples() for initial load of sample data for this prototype.
    Initializes session_state messages and chat_history to handle stateless nature of http calls.
    
    Parameters:
    None
    
    Returns:
    None

    """    

    st.set_page_config(page_title="Financial Advisor", page_icon="$€₹")
    st.title("Welcome to Financial Advisor!")    
    st.subheader("Ask me anything about financials")

    # Initialize session state for messages and chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    # If the session state is already initialized, print human and AI Bot messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input question
    if prompt := st.chat_input("Enter your query: "):
        msg = "Working on your query...."
        if "is_initialized" not in st.session_state:
            msg = "Initializing sample financial data for APPL, MSFT and TSLA. This will take couple of minutes. " + msg
        with st.spinner(msg):
            if "is_initialized" not in st.session_state:
                initialize_load_samples()       # Initialize and load sample data.
            response = query(question=prompt, chat_history=st.session_state.chat_history) 
            print(f"response Dict[str, Any] : {response}")            
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])    

            # Append user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.extend([(prompt, response["answer"])])
    
    # Side bar button to reset chat session.
    if st.sidebar.button("Reset chat history"):
        st.session_state.messages = []
        st.session_state.chat_history = []

    # Side bar button to load the sample data explicitly for demonstration of RAG. This happens automatically on first query as well.
    if st.sidebar.button("Load Sample Data"):
        with st.sidebar:
            if "is_initialized" not in st.session_state:
                with st.spinner("Loading Sample Data..."):
                    initialize_load_samples()
                st.success("Sample Data Loaded!")
            else:
                st.success("Samples Data already loaded!")

# Main program
if __name__ == "__main__":
    main_ux()