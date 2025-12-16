from dotenv import load_dotenv
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from load_vectorstore import download_faiss_index


class PredictionPipeline:
    def __init__(self):

        load_dotenv()
        print("Loaded .env")

        self.PROJECT_ID = os.environ.get("GCLOUD_PROJECT")
        self.GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
        self.GCS_INDEX_PATH = "vector_store"
        self.STORE_LOCAL_PATH = "/tmp/vector_store"

        self.book_df = pd.read_csv("final_book_df.csv")
        print("Book dataframe loaded")
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("Embedding model loaded")

        self.embedding_function = lambda text: self.embedding_model.encode(text,normalize_embeddings=True).tolist()

        self.vector_store = download_faiss_index(self.GCS_BUCKET_NAME,self.GCS_INDEX_PATH,self.STORE_LOCAL_PATH,self.PROJECT_ID, self.embedding_function)
        print("FAISS vector store loaded")

        self.reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Cross-encoder reranker loaded")

    def retrieve_semantic_recommendation(self,query: str,initial_top_k: int = 50,final_top_k: int = 25) -> list:

        candidates = self.vector_store.similarity_search(query,k=initial_top_k)

        if not candidates:
            return []

        input_pairs = [(query, doc.page_content) for doc in candidates]

        scores = self.reranker_model.predict(input_pairs,batch_size=16)

        ranked_docs = sorted(zip(candidates, scores),key=lambda x: x[1],reverse=True)[:final_top_k]

        isbn_list = [doc.metadata.get("isbn")for doc, _ in ranked_docs
            if doc.metadata.get("isbn") is not None ]

        return isbn_list
