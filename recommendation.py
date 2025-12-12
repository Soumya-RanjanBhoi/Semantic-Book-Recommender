import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

load_dotenv()


print("Starting to load Embedding model...")

embs = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding Model Loaded.")

print("Loading Vector Store...")
local_store = FAISS.load_local("vector_store", embs, allow_dangerous_deserialization=True)
print("Vector Store Loaded.")

print("Loading Reranker Model...")
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
reranker_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
print("Reranker Loaded.")


def retrieve_semantic_recommendation(query: str, initial_top_k: int = 50, final_top_k: int = 25) -> list:
    query = query.lower()
    
    candidates = local_store.similarity_search(query, k=initial_top_k)
    
    input_pairs = [[query, doc.page_content] for doc in candidates]
    
    relevance_scores = reranker_model.predict(input_pairs)
    
    scored_candidates = []
    for doc, score in zip(candidates, relevance_scores):
        scored_candidates.append({'doc': doc, 'score': score})

    scored_candidates.sort(key=lambda x: x['score'], reverse=True)

    final_recommendations = [item['doc'] for item in scored_candidates[:final_top_k]]
    
    isbn_list = []
    for doc in final_recommendations:
        try:
    
            isbn = doc.metadata.get('isbn', None) 
            if isbn:
                isbn_list.append(isbn)
        except Exception as e:
            print(f"Error extracting ISBN: {e}")

    return isbn_list