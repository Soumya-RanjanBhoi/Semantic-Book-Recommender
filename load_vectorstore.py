import os
from google.cloud import storage
from langchain_community.vectorstores import FAISS

def download_faiss_index(bucket_name,blob_path,local_dir,PROJECT_ID,embs):
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

        vector_store = FAISS.load_local(local_dir, embs, allow_dangerous_deserialization=True)
        return vector_store
    
    except Exception as e:
        print(f"Could Not download Vector Store from Cloud. Error-{e}")
        raise
