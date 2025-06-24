# retriever_setup.py

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient

QDRANT_PATH = "/Users/deakshshetty/Documents/RAG-based-role-access/embeddings/qdrant_local"
QDRANT_COLLECTION = "finrolebot"

def get_retriever():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Correct: create QdrantClient with local storage
    client = QdrantClient(path=QDRANT_PATH)

    # Now wrap the client using the LangChain Qdrant wrapper
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings,
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})
