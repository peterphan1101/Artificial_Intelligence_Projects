# TO DO: Testing the new load_to_vectdb_multimodals for different data type input

import os
import faiss
from dotenv import load_dotenv
from pymongo import MongoClient

from PIL import Image
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import CLIPProcessor, CLIPModel

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MongoDB connection string not found in environment variables.")

# Load Image 
class ImageLoader(BaseLoader):
    def __init__(self, image_path):
        self.image_path = image_path

    def load(self):
        image = Image.open(self.image_path) #  # Load the image using PIL & return as Langchain Document with image content
        return [Document(page_content="", image_content=image)]

class ImageEmbeddings: # Get image embeddings using CLIP
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_embeddings(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        return outputs.detach().numpy()

# 
class StringLoader(BaseLoader): # StringLoader to handle text data
    def __init__(self, string_content):
        self.string_content = string_content

    def load(self):
        return [Document(page_content=self.string_content)]

def load_to_vectdb_multimodal(company_cik, mongo_db, vector_db):
    """
    Load data of multiple modalities (text, images, etc.) from MongoDB, 
    generate embeddings, and store them in FAISS vector database.
    
    Parameters:
    - company_cik (str): Central Index Key (CIK) of the company
    - mongo_db (object): MongoDB connection object
    - vector_db (object): FAISS vector db connection object
    
    Returns:
    - vector_db (object): Updated FAISS vector db with new documents (multimodal)
    """
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200) # Embedding models and text splitter
    text_embedding_func = OpenAIEmbeddings()
    image_embedding_func = ImageEmbeddings()

    mongo_collection = mongo_db['FinDocs_text']
    company_cik = int(company_cik)
    financial_docs = mongo_collection.find({"cik": company_cik})

    for _fin_doc in financial_docs:
  
        fin_doc_content = _fin_doc.get("financial_doc", "") # text 
        if fin_doc_content:
            fin_txt_doc = StringLoader(fin_doc_content).load()
            fin_txt_data = text_splitter.split_documents(fin_txt_doc)
            vector_db.add_documents(fin_txt_data)

        
        image_path = _fin_doc.get("image_path") # image 
        if image_path:
            image_loader = ImageLoader(image_path)
            image_docs = image_loader.load()
            for img_doc in image_docs:
                image_embedding = image_embedding_func.get_embeddings(img_doc.image_content)
                vector_db.add_documents([Document(page_content="", embedding=image_embedding)])

    return vector_db

def initialize_multimodal_db():
    """
    Initializes the connection to MongoDB, OpenAI LLM embeddings for text, 
    and FAISS vector database for storing multimodal data.
    """
    embd_func = OpenAIEmbeddings() # embeddings and FAISS vector DB
    embd_dimension = 1536  # Using text-embedding-ada-002 for text
    docstore = InMemoryDocstore()
    
    vector_db = FAISS( # FAISS index for Euclidean distance
        embedding_function=embd_func,
        index=faiss.IndexFlatL2(embd_dimension),
        docstore=docstore,
        index_to_docstore_id={},
        relevance_score_fn=None,
        normalize_L2=False,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    
    mongo_client = MongoClient(MONGO_URI) # MongoDB client
    mongo_db = mongo_client['llm_mongo_db']

    return mongo_db, vector_db

if __name__ == "__main__":
    mongo_db, vector_db = initialize_multimodal_db() # Initialize connections and vector db

    # Load multimodal data for a specific company (example CIK)
    company_cik = "0000320193"  # Example for Apple Inc.
    vector_db = load_to_vectdb_multimodal(company_cik, mongo_db, vector_db)
    faiss.write_index(vector_db.index, "multimodal_faiss_index.index")  # Save the FAISS index for future retrievals
