import os
import random
import time
import openai
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from pymongo import MongoClient
import faiss
import matplotlib.pyplot as plt  # Added for plotting

# LangChain Modules
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from langchain.docstore import InMemoryDocstore

# Evaluation metrics from RAGAS
from ragas import evaluate, RunConfig
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

# Custom modules
from load_companies import (
    load_companies_to_mongo,
    load_companies_to_vectdb,
    read_sample_companies,
)

# Load environment vars from `.env` file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MongoDB connection string not found in environment variables.")

samples_file = "sample_companies.yml"

def initialize_components():
    """
    Initializes the language model, embeddings, vector store, and loads company data into MongoDB and the vector store.

    Returns:
        tuple: A tuple containing the initialized ChatOpenAI model and the FAISS vector store.
    """
    
    llm_openai = ChatOpenAI(model="gpt-3.5-turbo") # language model (LLM) 3.5

    embd_func = OpenAIEmbeddings()
    embd_dimension = 1536  # Dimension of OpenAI embeddings

    docstore = InMemoryDocstore({}) # in-memory document store
 
    index = faiss.IndexFlatL2(embd_dimension) # FAISS vector store
    vector_db = FAISS(
        embedding_function=embd_func,
        index=index,
        docstore=docstore,
        index_to_docstore_id={},
        relevance_score_fn=None,
        normalize_L2=False,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    # Connect to MongoDB
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client['llm_mongo_db']
    mongo_collection = mongo_db['FinDocs_text']
    mongo_collection.drop()  # Clear existing data

    # Load sample companies and data into MongoDB and vector store
    sample_companies = read_sample_companies(samples_file)
    load_companies_to_mongo(sample_companies, mongo_db)
    load_companies_to_vectdb(sample_companies, mongo_db, vector_db)

    return llm_openai, vector_db

llm_openai, vector_db = initialize_components()

rag_chain_openai = ConversationalRetrievalChain.from_llm( # Retrieval Augmented Generation (RAG) chain
    llm=llm_openai,
    retriever=vector_db.as_retriever()
)

def generate_test_dataset(num_samples=10):
    """
    Generates a test dataset with questions and ground truth answers.

    Args:
        num_samples (int): Number of samples to generate.

    Returns:
        Dataset: A Hugging Face Dataset object containing the test questions and ground truths.
    """
    queries = [
        "What are Apple's strategies for innovation in the tech industry?",
        "How has Apple's financial performance evolved over the last decade?",
        "What are the key revenue streams for Apple?",
        "How does Apple manage supply chain risks?",
        "What are Microsoft's growth strategies in the cloud computing sector?",
        "How has Microsoft's acquisition strategy impacted its market position?",
        "What role does Microsoft play in the enterprise software market?",
        "How does Microsoft ensure sustainability in its operations?",
        "How does Tesla's production efficiency impact its profitability?",
        "What are Tesla's strategies for expanding its market share globally?",
        "How does Tesla approach innovation in electric vehicles?",
        "What are the financial challenges faced by Tesla in scaling production?",
    ]
    dataset = {"question": [], "ground_truth": []}
    for _ in range(num_samples):
        question = random.choice(queries)
        relevant_docs = vector_db.as_retriever().get_relevant_documents(question)
        ground_truth = relevant_docs[0].page_content if relevant_docs else ""
        dataset["question"].append(question)
        dataset["ground_truth"].append(ground_truth)
    return Dataset.from_dict(dataset)


testset = generate_test_dataset(num_samples=10) # Test dataset

def run_evaluation(llm_name, llm, rag_chain, testset):
    """
    Runs evaluation on the specified LLM using the RAG chain and calculates metrics.

    Args:
        llm_name (str): Name of the language model.
        llm (LLM): The language model instance.
        rag_chain (ConversationalRetrievalChain): The RAG chain to use.
        testset (Dataset): The test dataset.

    Returns:
        DataFrame: A pandas DataFrame containing the evaluation results.
    """
    answers = []
    contexts = []
    for query in testset["question"]:
        print(f"Processing query for {llm_name}: {query}")
        try:
            response = rag_chain({"question": query, "chat_history": []})
            formatted_result = response.get('answer', "No answer")
            print(f"{llm_name} answer: {formatted_result}")
            answers.append(formatted_result)
        except Exception as e:
            print(f"Error invoking chain for query '{query}': {e}")
            answers.append("Error")
        try:
            relevant_docs = vector_db.as_retriever().get_relevant_documents(query)
            contexts.append([doc.page_content for doc in relevant_docs] or [""])
        except Exception as e:
            print(f"Error retrieving documents for query '{query}': {e}")
            contexts.append([""])

    data = {
        "question": testset["question"],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": testset["ground_truth"],
    }
    dataset = Dataset.from_dict(data)

    run_config = RunConfig(max_workers=2, timeout=50) # RAGAS metrics evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=llm,
        embeddings=None,
        run_config=run_config
    )

    df_eval = result.to_pandas()
    df_eval['model'] = llm_name
    print(df_eval)
    return df_eval

results_openai = run_evaluation("GPT-3.5-turbo", llm_openai, rag_chain_openai, testset) # Model evaluation

csv_file = "llm_evaluation_results_RAG.csv"
results_openai.to_csv(csv_file, index=False)
print(f"Evaluation results saved to {csv_file}")

def plot_evaluation_results(df_eval):
    """
    Plots the evaluation metrics and saves the graphs.

    Args:
        df_eval (DataFrame): The DataFrame containing evaluation metrics.

    Returns:
        None
    """
    metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        df_eval[metric].plot(kind='bar')
        plt.title(f'{metric.capitalize()} for {df_eval["model"].iloc[0]}')
        plt.xlabel('Test Samples')
        plt.ylabel(metric.capitalize())
        plt.tight_layout()
        graph_filename = f'{metric}_{df_eval["model"].iloc[0]}.png'
        plt.savefig(graph_filename)
        print(f"Graph saved as {graph_filename}")
        plt.close()

plot_evaluation_results(results_openai)
