import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

# ---- CONFIG ---- #
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DOC_PATH = "/Users/deakshshetty/Documents/RAG-based-role-access/DS-RPC-01/data"
QDRANT_COLLECTION = "finrolebot"
QDRANT_PATH = "/Users/deakshshetty/Documents/RAG-based-role-access/embeddings/qdrant_local"
# ---------------- #

def load_and_split_documents(doc_path=BASE_DOC_PATH):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for role_dir in os.listdir(doc_path):
        role_path = os.path.join(doc_path, role_dir)
        if not os.path.isdir(role_path):
            continue

        for file_name in os.listdir(role_path):
            if not file_name.endswith(".md"):
                continue

            file_path = os.path.join(role_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = splitter.split_text(text)
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk:  # Only add if chunk is non-empty after stripping
                    # Debug print
                    print(f"Adding chunk from {file_name} with length {len(chunk)}")
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_name,
                            "role": role_dir.lower()
                        }
                    )
                    all_docs.append(doc)

    return all_docs

def store_in_qdrant(documents):
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    client = QdrantClient(path=QDRANT_PATH)

    # Delete collection if it exists - start fresh to avoid old corrupt data
    if client.collection_exists(QDRANT_COLLECTION):
        print(f"Deleting existing collection '{QDRANT_COLLECTION}'...")
        client.delete_collection(QDRANT_COLLECTION)

    # Create collection fresh
    dim = embedding_model.client.get_sentence_embedding_dimension()
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Created collection '{QDRANT_COLLECTION}' with dimension {dim}.")

    # Prepare data for upsert, include page_content in payload!
    points = [
        PointStruct(
            id=i,
            vector=embedding_model.embed_documents([documents[i].page_content])[0],
            payload={
                **documents[i].metadata,
                "page_content": documents[i].page_content
            }
        )
        for i in range(len(documents))
    ]

    # Upsert points
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )
    print(f"Upserted {len(points)} points into Qdrant collection '{QDRANT_COLLECTION}'.")

if __name__ == "__main__":
    print("📄 Loading and splitting documents...")
    docs = load_and_split_documents()
    print(f"✅ Loaded and split {len(docs)} text chunks.")

    print("📥 Indexing into Qdrant...")
    store_in_qdrant(docs)
    print("✅ Documents indexed into Qdrant vector store.")
