import os
import faiss
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai.embeddings.base import OpenAIEmbeddings

from pymongo import MongoClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MongoDB connection string not found in environment variables.")

class StringLoader(BaseLoader):
    def __init__(self, string_content):
        self.string_content = string_content

    def load(self):
        return [Document(page_content=self.string_content)]
    
def load_to_vectdb(company_cik, mongo_db, vector_db):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap=200
    )
    mongo_collection = mongo_db['FinDocs_text']
    company_cik = int(company_cik)
    financial_docs = mongo_collection.find({"cik":company_cik})
    for _fin_doc in financial_docs:
        fin_doc_content = _fin_doc["financial_doc"]
        fin_txt_doc = StringLoader(fin_doc_content).load()
        fin_txt_data = text_splitter.split_documents(fin_txt_doc)
        vector_db.add_documents(fin_txt_data)
    return vector_db