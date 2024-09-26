import os
from dotenv import load_dotenv
import streamlit as st

# Updated imports from the standard 'langchain' library
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_retriever_tool, create_conversational_retrieval_agent
from langchain.tools import BaseTool, Tool
from langchain.schema import ChatMessage

from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.utilities import SerpAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


load_dotenv() # Load environment variables from `.env` file

openai_api_key = os.environ['OPENAI_API_KEY'] # Retrieve API keys from environment variables
serpapi_api_key = os.environ['SERPAPI_API_KEY']

st.set_page_config(page_title="Financial Advisor", page_icon="$€₹") # Streamlit page config
st.header('Welcome to Personal Financial Advisor!')
 
search = SerpAPIWrapper() # SerpAPI search tool

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

appl_10k = PyPDFLoader('AAPL_0000320193.pdf').load() # Financial Doc
fin_documents = text_splitter.split_documents(appl_10k)

db = FAISS.from_documents(fin_documents, OpenAIEmbeddings()) # Vector Store using FAISS

memory = ConversationBufferMemory( # Conversation Memory
    return_messages=True,
    memory_key="chat_history",
    output_key="output"
)

llm = ChatOpenAI()

tools = [ # Agent tool
    Tool.from_function( 
        func=search.run,
        name="Search",
        description="Useful for answering questions about current events."
    ),
    create_retriever_tool(
        db.as_retriever(),
        name="FinancialData",
        description="Searches and returns financial documents."
    )
]

agent = create_conversational_retrieval_agent( # Conversational Retrieval Agent
    llm=llm,
    tools=tools,
    memory=memory,
    verbose=True
)

agent = create_conversational_retrieval_agent( # Conversational retrieval agent with memory
    llm=llm,
    tools=tools,
    memory=memory,
    verbose=True
)


user_query = st.text_input( # User input prompt
    "**How can I help you with financial information?**",
    placeholder="Ask me anything!"
)


if "messages" not in st.session_state: # Initialize session state for messages and memory
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "How can I help you with financial information?"
    }]
if "memory" not in st.session_state:
    st.session_state['memory'] = memory

for msg in st.session_state["messages"]: # Chat history
    st.chat_message(msg["role"]).write(msg["content"])

def display_msg(msg, author):
    """
    Function to display a message in the chat interface and append it to the session state.

    Parameters:
    - msg (str): The message content to display.
    - author (str): The role of the message sender ('user' or 'assistant').
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

# Handle user query
if user_query:
    display_msg(user_query, 'user')
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container()) # Initialize Streamlit callback handler for real-time feedback
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response}) # Assistant's response to the session state
        st.write(response)


if st.sidebar.button("Reset chat history"): # Reset chat history from the sidebar
    st.session_state.messages = []
