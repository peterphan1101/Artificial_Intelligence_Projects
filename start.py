import os
from dotenv import load_dotenv
import streamlit as st
import faiss

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain.tools import BaseTool, Tool, tool
from langchain.schema import ChatMessage

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from langchain_openai.llms.base import OpenAI
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
from load_companies import load_companies_to_mongo, load_companies_to_vectdb, read_sample_companies

from pymongo import MongoClient

## Credit : Coding patterns are inspired from below Oreilly's LLM course and book examples:
## Gen AI - RAG Application Development using LangChain By Manas Dasgupta
## Building LLM Powered Applications By Valentina Alto

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MongoDB connection string not found in environment variables.")

openai_api_key = os.environ['OPENAI_API_KEY']
serpapi_api_key = os.environ['SERPAPI_API_KEY']
samples_file = "sample_companies.yml"

def query(question, chat_history):

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
        sample_companies = read_sample_companies(samples_file)
        load_companies_to_mongo(sample_companies, mongo_db)
        load_companies_to_vectdb(sample_companies, mongo_db, vector_db)
        st.session_state.vector_db = vector_db
        st.session_state.llm = llm
        st.session_state.mongo_db = mongo_db
        st.session_state.is_initialized = "True"
    
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

def main_ux():

    st.set_page_config(page_title="Financial Advisor", page_icon="$€₹")
    st.title("Welcome to Personal Financial Advisor!")    
    st.subheader("Ask me anything about financials")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter your query: "):
        with st.spinner("Working on your query...."):
            response = query(question=prompt, chat_history=st.session_state.chat_history)            
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])    

            # Append user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.extend([(prompt, response["answer"])])
    
    if st.sidebar.button("Reset chat history"):
        st.session_state.messages = []

# Main program
if __name__ == "__main__":
    main_ux() 
    