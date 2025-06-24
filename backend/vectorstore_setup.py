# backend/vectorstore_setup.py

import os
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue

QDRANT_COLLECTION = "finrolebot"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_vectorstore_for_role(role: str):
    role = role.lower()

    # C-level roles get access to all documents (no filter)
    c_level_roles = ["ceo", "cxo", "chief executive officer", "chief", "executive", "c-level", "C-LEVEL"]

    if role in c_level_roles:
        filter_ = None
    else:
        # For HR, allow access to hr + general
        if role == "hr":
            allowed_roles = ["hr", "general"]
        else:
            allowed_roles = [role]

        # Build filter condition
        if len(allowed_roles) == 1:
            filter_ = Filter(
                must=[
                    FieldCondition(
                        key="metadata.role",
                        match=MatchValue(value=allowed_roles[0])
                    )
                ]
            )
        else:
            filter_ = Filter(
                must=[
                    FieldCondition(
                        key="metadata.role",
                        match=MatchValue(value=allowed_roles)
                    )
                ]
            )

    print("🔎 Role Filter:", filter_)
    return get_vectorstore(filter_metadata=filter_)


def get_vectorstore(filter_metadata=None):
    # Try to use Qdrant Cloud if ENV vars are set, else fallback to local
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if QDRANT_URL and QDRANT_API_KEY:
        print("🚀 Using Qdrant Cloud client")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        print("🧪 Using local Qdrant client (tempfile path)")
        QDRANT_PATH = tempfile.gettempdir() + "/qdrant"
        client = QdrantClient(path=QDRANT_PATH)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectordb = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embedding_model,
    )

    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": filter_metadata
        }
    )

    return retriever
