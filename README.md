# 📚 Semantic Book Recommender System

A high-precision book recommendation engine that uses **Deep Learning** and **Vector Search** to understand the true meaning behind book descriptions. This system goes beyond keyword matching to find books based on themes, moods, and complex narratives.



---

## 🚀 Overview
Most book search engines rely on simple keyword matching. This project implements a **Two-Stage Retrieval Pipeline** to provide highly accurate results:

1.  **Stage 1: Semantic Retrieval (Bi-Encoder)** - Quickly narrows down thousands of books to the top 50 candidates using **FAISS** and dense vector embeddings.
2.  **Stage 2: Precision Reranking (Cross-Encoder)** - Re-scores the top candidates by analyzing the query and description simultaneously, filtering out "semantic noise" and irrelevant genres.

## 🛠️ Tech Stack
* **Python 3.9+**
* **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
* **LLM Framework:** [LangChain](https://www.langchain.com/)
* **Embedding Model (Bi-Encoder):** `sentence-transformers/all-MiniLM-L6-v2`
* **Reranker Model (Cross-Encoder):** `cross-encoder/ms-marco-MiniLM-L-6-v2`

## 📂 Data Pipeline
The system processes raw text files in the format `ISBN13 Description`. 
To ensure high-quality embeddings, the system performs an automated cleaning step:
* **Extraction:** Strips the 13-digit ISBN from the start of the text.
* **Metadata Storage:** Stores the ISBN in a metadata dictionary.
* **Clean Embedding:** Only the actual book description is used for vector generation, preventing numerical noise (ISBNs) from skewing search results.

## ⚙️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Soumya-RanjanBhoi/semantic-book-recommender.git
   cd semantic-book-recommender
   ```
2. ** Install Dependencies **
  ```bash 
    pip install -r requirements.txt
```
3.RUN 
```bash
  python dashboard.py
```
