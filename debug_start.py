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

st.set_page_config(page_title="Financial Advisor", page_icon="$€₹")
st.header('Welcome to Personal Financial Advisor!')

search = SerpAPIWrapper()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap=200
)

appl_10k = PyPDFLoader('APPL_0000320193.pdf').load()
appl_fin_data = text_splitter.split_documents(appl_10k)

fin_documents = text_splitter.split_documents(appl_fin_data)
db = FAISS.from_documents(fin_documents, OpenAIEmbeddings())

memory = ConversationBufferMemory(
    return_messages=True, 
    memory_key="chat_history", 
    output_key="output"
)

llm = ChatOpenAI()
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
    ),
    create_retriever_tool(
        db.as_retriever(), 
        "financial_data",
        "Searches and returns financial documents"
    )
    ]

agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)

user_query = st.text_input(
    "**How can I help you with financial information?**",
    placeholder="Ask me anything!"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with financial information?"}]
if "memory" not in st.session_state:
    st.session_state['memory'] = memory


for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


def display_msg(msg, author):

    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

if user_query:
    display_msg(user_query, 'user')
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

if st.sidebar.button("Reset chat history"):
    st.session_state.messages = []