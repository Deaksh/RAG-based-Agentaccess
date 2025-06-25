# backend/vectorstore_setup.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
from dotenv import load_dotenv
load_dotenv()

QDRANT_COLLECTION = "finrolebot"
QDRANT_URL = os.getenv("QDRANT_URL")  # Set this in Streamlit Cloud Secrets or your .env
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_vectorstore_for_role(role: str):
    role = role.lower()
    c_level_roles = ["ceo", "cxo", "chief executive officer", "chief", "executive","c-level","C-LEVEL"]

    if role in c_level_roles:
        filter_ = None
    else:
        allowed_roles = ["hr", "general"] if role == "hr" else [role]

        filter_ = Filter(
            must=[
                FieldCondition(
                    key="metadata.role",
                    match=MatchValue(value=allowed_roles[0])
                )
            ]
        ) if len(allowed_roles) == 1 else Filter(
            must=[
                FieldCondition(
                    key="metadata.role",
                    values=allowed_roles
                )
            ]
        )

    print("🔎 Role Filter:", filter_)
    return get_vectorstore(filter_metadata=filter_)

def get_vectorstore(filter_metadata=None):
    from qdrant_client.models import VectorParams

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],           # Set via Streamlit secrets
        api_key=os.environ["QDRANT_API_KEY"]    # Set via Streamlit secrets
    )

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Embedding dimension for MiniLM is 384
    vector_size = 384

    # Check if collection exists
    existing_collections = [col.name for col in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing_collections:
        print(f"🛠 Creating collection: {QDRANT_COLLECTION}")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance="Cosine",
            ),
        )

    vectordb = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION,
    embedding=embedding_model,
    vector_name="default",  # ✅ Add this line
  )

    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": filter_metadata
        }
    )

    return retriever
