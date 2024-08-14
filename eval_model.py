import os
import random
import openai
import pandas as pd
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
from ragas import evaluate, RunConfig
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from anthropic import Client, HUMAN_PROMPT, AI_PROMPT
from openai import OpenAI
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Any, List, Optional
from pydantic import Field

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
llama_api_key = os.getenv("LLAMA_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MongoDB connection string not found in environment variables.")
samples_file = "sample_companies.yml"

def initialize_components():
    llm_openai = ChatOpenAI(model="gpt-3.5-turbo")
    # anthropic_client = Client(api_key=anthropic_api_key)
    # llama_client = OpenAI(api_key=llama_api_key, base_url="https://api.llama-api.com")
    # gemini_client = None
    # try:
    #     genai.configure(api_key=gemini_api_key)
    #     gemini_client = genai
    #     print("Gemini client successfully configured.")
    # except Exception as e:
    #     print(f"Failed to configure Gemini client: {e}")
    embd_func = OpenAIEmbeddings()
    embd_dimension = 1536
    docstore = InMemoryDocstore()
    vector_db = FAISS(embedding_function=embd_func, index=faiss.IndexFlatL2(embd_dimension), docstore=docstore, index_to_docstore_id={}, relevance_score_fn=None, normalize_L2=False, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client['llm_mongo_db']
    mongo_collection = mongo_db['FinDocs_text']
    mongo_collection.drop()
    sample_companies = read_sample_companies(samples_file)
    load_companies_to_mongo(sample_companies, mongo_db)
    load_companies_to_vectdb(sample_companies, mongo_db, vector_db)
    return llm_openai, vector_db # anthropic_client, llama_client, gemini_client

llm_openai, vector_db = initialize_components() #anthropic_client,  llama_client, gemini_client
# Initialize a rag_chain for testing
rag_chain_openai = ConversationalRetrievalChain.from_llm(
    llm=llm_openai,
    retriever=vector_db.as_retriever()
)

# class ClaudeLLM(LLM):
#     client: Any = Field(default=None, exclude=True)

#     def __init__(self, client: Any):
#         super().__init__(client=client)
    
#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         response = self.client.completions.create(
#             model="claude-v1",
#             prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
#             max_tokens_to_sample=300
#         )
#         return response.completion.strip()

#     @property
#     def _llm_type(self) -> str:
#         return "claude"
    
# class LlamaLLM(LLM):
#     client: Any = Field(default=None, exclude=True)

#     def __init__(self, client: Any):
#         super().__init__(client=client)

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         response = self.client.chat.completions.create(
#             model="llama-13b-chat",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=300
#         )
#         return response.choices[0].message.content.strip()

#     @property
#     def _llm_type(self) -> str:
#         return "llama"

# class GeminiLLM(LLM):
#     client: Any = Field(default=None, exclude=True)

#     def __init__(self, client: Any):
#         super().__init__(client=client)

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         if not self.client:
#             raise RuntimeError("Gemini client is not configured.")
#         response = self.client.generate_text(
#             model="gemini-v1",
#             prompt=prompt,
#             max_tokens=300
#         )
#         return response['candidates'][0]['output'].strip() 

#     @property
#     def _llm_type(self) -> str:
#         return "gemini"


# claude_llm = ClaudeLLM(anthropic_client)
# llama_llm = LlamaLLM(llama_client)
# gemini_llm = GeminiLLM(gemini_client)

# rag_chain_claude = ConversationalRetrievalChain.from_llm(
#     llm=claude_llm,
#     retriever=vector_db.as_retriever()
# )
# rag_chain_llama = ConversationalRetrievalChain.from_llm(
#     llm=llama_llm,
#     retriever=vector_db.as_retriever()
# )
# rag_chain_gemini = ConversationalRetrievalChain.from_llm(
#     llm=gemini_llm,
#     retriever=vector_db.as_retriever()
# )
  
# test dataset with company specific questions
def generate_test_dataset(num_samples=10):
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
    dataset = {"question": [], "ground_truth": [] }
    for _ in range(num_samples):
        question = random.choice(queries)
        relevant_docs = vector_db.as_retriever().invoke(question)
        dataset["question"].append(question)
        dataset["ground_truth"].append(relevant_docs[0].page_content if relevant_docs else "")
    return Dataset.from_dict(dataset)

# Generate test dataset
testset = generate_test_dataset(num_samples=10)

def run_evaluation(llm_name, llm, rag_chain, testset):
    answers = []
    contexts = []
    for query in testset["question"]:
        print("Processing query for: ", llm_name, ": ", query)
        try:
            response = rag_chain.invoke({"question": query, "chat_history": []})
            formatted_result = response.get('answer', "No answer")
            print(llm_name, "answer: ", formatted_result)
            answers.append(formatted_result)
        except Exception as e:
            print("Error invoking chain for query ", query, ": ", e)
            answers.append("Error")
        try:
            relevant_docs = vector_db.as_retriever().invoke(query)
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

    # Evaluate using RAGAS metrics
    run_config = RunConfig(max_workers=2, timeout=50)
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

# # Evaluate each model with RAG
results_openai = run_evaluation("GPT-3.5-turbo", llm_openai, rag_chain_openai, testset)
# results_claude = run_evaluation("Claude", claude_llm, rag_chain_claude, testset)
# results_llama = run_evaluation("Llama", llama_llm, rag_chain_llama, testset)
# results_gemini = run_evaluation("Gemini", gemini_llm, rag_chain_gemini, testset)
comparison_df = pd.concat([results_openai])
print(comparison_df)
csv_file = "llm_evaluation_results_RAG.csv"
comparison_df.to_csv(csv_file, index=False)
