import os
import random
import openai
import pandas as pd
import time
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
from load_companies import load_companies_to_mongo, load_companies_to_vectdb, read_sample_companies
from datasets import Dataset
from pymongo import MongoClient
import faiss

# Get environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MongoDB connection string not found in environment variables.")

samples_file = "sample_companies.yml"

# Initialize llm and vector embeddings for evaluation
def initialize_components():
    llm_openai = ChatOpenAI(model="gpt-3.5-turbo")
    embd_func = OpenAIEmbeddings()
    embd_dimension = 1536
    docstore = InMemoryDocstore()
    vector_db = FAISS(embedding_function=embd_func, index=faiss.IndexFlatL2(embd_dimension),
                      docstore=docstore, index_to_docstore_id={}, relevance_score_fn=None,
                      normalize_L2=False, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client['llm_mongo_db']
    mongo_collection = mongo_db['FinDocs_text']
    mongo_collection.drop()
    sample_companies = read_sample_companies(samples_file)
    load_companies_to_mongo(sample_companies, mongo_db)
    load_companies_to_vectdb(sample_companies, mongo_db, vector_db)
    
    return llm_openai, vector_db

llm_openai, vector_db = initialize_components()
# Initialize rag chain for RAG evaluation
rag_chain_openai = ConversationalRetrievalChain.from_llm(
    llm=llm_openai,
    retriever=vector_db.as_retriever()
)

# Helper function for evaluation test set
def generate_test_dataset(num_samples=40):
    queries = [
    "What are the latest technology trends that Apple is currently exploring?",
    "How does Apple integrate artificial intelligence into its products?",
    "What measures has Apple taken to secure data privacy in its devices?",
    "How does Apple's brand loyalty affect its product development strategy?",
    "What partnerships has Apple formed to advance its innovation?",
    "How effective are Apple’s marketing strategies in emerging markets?",
    "What steps is Apple taking to enhance its services division?",
    "How does Apple foster innovation within its corporate culture?",
    "What impact has the shift to remote work had on Apple's product line?",
    "How is Apple addressing the competitive threat from Android devices?",
    "What recent cloud technology innovations has Microsoft developed?",
    "How is Microsoft leveraging AI to enhance its cloud services?",
    "What new markets is Microsoft targeting with its enterprise solutions?",
    "How does Microsoft's R&D spending compare to its tech industry peers?",
    "What initiatives has Microsoft taken to partner with governments and NGOs?",
    "How has Microsoft adapted its products for the education sector?",
    "What are the challenges Microsoft faces in data security for cloud services?",
    "How is Microsoft positioning itself against competitors like Amazon AWS and Google Cloud?",
    "What role does customer feedback play in Microsoft's software development?",
    "How has Microsoft's focus on enterprise customers evolved over the years?",
    "What recent advancements has Tesla made in battery technology?",
    "How is Tesla improving its autopilot features?",
    "What strategies does Tesla use to manage its global supply chain?",
    "How does Tesla’s entry into renewable energy products affect its brand?",
    "What are the challenges Tesla faces with regulatory compliance worldwide?",
    "How does Tesla balance innovation with sustainability?",
    "What are Tesla’s strategies for managing workforce during rapid expansion?",
    "How is Tesla addressing concerns related to driver safety in its vehicles?",
    "What partnerships is Tesla forming to secure raw materials for batteries?",
    "How does Tesla's market strategy differ in the U.S. versus China?",
    "How are Apple, Microsoft, and Tesla promoting sustainability in their operations?",
    "What are the key competitive advantages of Apple, Microsoft, and Tesla?",
    "How do Apple, Microsoft, and Tesla handle international trade tensions?",
    "What are the major risks associated with investing in Apple, Microsoft, and Tesla?",
    "How have Apple, Microsoft, and Tesla's stock prices been influenced by current economic conditions?",
    "What role do Apple, Microsoft, and Tesla play in the future of artificial intelligence?",
    "How do these companies ensure compliance with global privacy laws?",
    "What impact do political changes in the U.S. have on these companies?",
    "How are these companies planning for post-pandemic economic shifts?",
    "What are the implications of antitrust investigations on Apple, Microsoft, and Tesla?"
]
    dataset = {"question": [],"ground_truth": []}
    for _ in range(num_samples):
        question = random.choice(queries)
        relevant_docs = vector_db.as_retriever().invoke(question)
        ground_truth = relevant_docs[0].page_content if relevant_docs else ""
        dataset["question"].append(question)
        dataset["ground_truth"].append(ground_truth)
    return Dataset.from_dict(dataset)

# Get test question set with 40 questions
testset = generate_test_dataset(num_samples=40)

def measure_response_times(llm_name, rag_chain, testset, use_rag=True):
    response_times = []

    for query in testset["question"]:
        start_time = time.time()
        # Evaluation of response time based on Retreival augmented and non Retreival augmented generation
        if use_rag: # using rag
            rag_chain.invoke({"question": query, "chat_history": []})
        else: # don't use rag
            llm_openai(query)
        end_time = time.time()
        response_time = end_time - start_time
        response_times.append(response_time)
        print(f"{llm_name} {'with' if use_rag else 'without'} RAG - Query: {query} - Response Time: {response_time:.4f} seconds")
    return response_times

# Evaluations made on GPT 3.5 Turbo (api calls this llm)
response_times_with_rag = measure_response_times("GPT-3.5-turbo", rag_chain_openai, testset, use_rag=True)
response_times_without_rag = measure_response_times("GPT-3.5-turbo", rag_chain_openai, testset, use_rag=False)
response_df = pd.DataFrame({
    "Question": testset["question"],
    "Response Time (With RAG)": response_times_with_rag,
    "Response Time (Without RAG)": response_times_without_rag
})
# Calculate metrics with response times
average_with_rag = response_df["Response Time (With RAG)"].mean()
average_without_rag = response_df["Response Time (Without RAG)"].mean()
print("\nAverage Response Time (With RAG): {:.4f} seconds".format(average_with_rag))
print("Average Response Time (Without RAG): {:.4f} seconds".format(average_without_rag))
response_df.to_csv("response_time_analysis.csv", index=False)
print(f"Response time analysis saved to response_time_analysis.csv")
