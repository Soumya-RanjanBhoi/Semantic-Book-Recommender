import os
from google.cloud import storage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


print("Starting to load Embedding model...")

embs = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding Model Loaded.")

PROJECT_ID=os.environ.get("GCLOUD_PROJECT")
GCS_BUCKET_NAME=os.environ.get("GCS_BUCKET_NAME")
GCS_INDEX_PATH="vector_store"
LOCAL_PATH= "/tmp/vector_store"

def download_faiss_index(bucket_name,blob_path,local_dir):

    try:
        if not os.path.exists(local_dir):
            os.makedirs(local_dir,exist_ok=True)

        storage_client = storage.Client(project=PROJECT_ID)
        bucket= storage_client.bucket(bucket_name)

        blob_faiss = bucket.blob(f"{blob_path}/index.faiss")
        blob_faiss.download_to_filename(f"{local_dir}/index.faiss")

        blob_pkl = bucket.blob(f"{blob_path}/index.pkl")
        blob_pkl.download_to_filename(f"{local_dir}/index.pkl")

        print("Successfully Downloaded Faiss Vector Store")

        print(f"Downloaded index files to {local_dir}")
    
    except Exception as e:
        print(f"Could Not download Vector Store from Cloud. Error-{e}")


download_faiss_index(GCS_BUCKET_NAME, GCS_INDEX_PATH, LOCAL_PATH)
local_store = FAISS.load_local(LOCAL_PATH, embs, allow_dangerous_deserialization=True)
print("Loaded Vector Store Successfully")

