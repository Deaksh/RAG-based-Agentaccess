# backend/vectorstore_setup.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, VectorParams, PayloadSchemaType
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_COLLECTION = "finrolebot"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ✅ canonical name

def get_vectorstore_for_role(role: str):
    role = role.lower()
    c_level_roles = ["ceo", "cxo", "chief executive officer", "chief", "executive", "c-level"]

    if role in c_level_roles:
        filter_ = None
    else:
        allowed_roles = ["hr", "general", "employee"] if role == "hr" else [role]

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
                        match=MatchAny(any=allowed_roles)
                    )
                ]
            )

    print("🔎 Role Filter:", filter_)
    return get_vectorstore(filter_metadata=filter_)


def get_vectorstore(filter_metadata=None):
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_size = 384  # all-MiniLM-L6-v2

    existing_collections = [col.name for col in client.get_collections().collections]
    collection_exists = QDRANT_COLLECTION in existing_collections

    if not collection_exists:
        print(f"🛠 Creating collection: {QDRANT_COLLECTION}")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance="Cosine")
        )

    # Ensure payload index for metadata.role
    payload_schema = getattr(client.get_collection(QDRANT_COLLECTION), "payload_schema", {})
    if "metadata.role" not in payload_schema:
        print("🔧 Creating payload index for metadata.role...")
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="metadata.role",
            field_schema=PayloadSchemaType.KEYWORD
        )

    vectordb = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embedding_model,
        vector_name="",  # default unnamed vectors
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4, "filter": filter_metadata}
    )

    return retriever
