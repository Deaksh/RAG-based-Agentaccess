# backend/vectorstore_setup.py

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
import tempfile
from qdrant_client import QdrantClient

QDRANT_COLLECTION = "finrolebot"
QDRANT_PATH = "/Users/deakshshetty/Documents/RAG-based-role-access/embeddings/qdrant_local"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


from qdrant_client.models import Filter, FieldCondition, MatchValue

def get_vectorstore_for_role(role: str):
    role = role.lower()

    # C-level roles get access to all documents (no filter)
    c_level_roles = ["ceo", "cxo", "chief executive officer", "chief", "executive","c-level","C-LEVEL"]

    if role in c_level_roles:
        filter_ = None  # No filter means access all
    else:
        # For HR, allow access to hr + general
        if role == "hr":
            allowed_roles = ["hr", "general"]
        else:
            allowed_roles = [role]

        # Build filter with 'must' containing multiple roles if needed
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
                        values=allowed_roles
                    )
                ]
            )

    print("🔎 Role Filter:", filter_)
    return get_vectorstore(filter_metadata=filter_)



def get_vectorstore(filter_metadata=None):
    QDRANT_PATH = tempfile.gettempdir() + "/qdrant"  # safe path for Streamlit Cloud
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
