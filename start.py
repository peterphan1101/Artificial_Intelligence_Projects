import os
from dotenv import load_dotenv
import streamlit as st

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

load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']
serpapi_api_key = os.environ['SERPAPI_API_KEY']

def query(question, chat_history):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap=200
    )

    appl_10k = PyPDFLoader('APPL_0000320193.pdf').load()
    appl_fin_data = text_splitter.split_documents(appl_10k)

    fin_documents = text_splitter.split_documents(appl_fin_data)
    new_db = FAISS.from_documents(fin_documents, OpenAIEmbeddings())

    memory = ConversationBufferMemory(
        return_messages=True, 
        memory_key="chat_history", 
        output_key="output"
    )

    llm = ChatOpenAI(temperature=0)

    query = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=new_db.as_retriever(), 
        return_source_documents=True)
    return query({"question": question, "chat_history": chat_history})


def show_ux():
    
    st.set_page_config(page_title="Financial Advisor", page_icon="$€₹")
    st.title("Welcome to Personal Financial Advisor!")    
    st.subheader("Ask me anything about financials")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun
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
    show_ux() 
    