import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Path to your documents folder
DATA_DIR = "/Users/deakshshetty/Documents/RAG-based-role-access/DS-RPC-01/data"
QDRANT_PATH = "./embeddings/qdrant_local"
QDRANT_COLLECTION = "finrolebot"

# 🧠 Embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Function to extract role from the file path
def extract_role_from_path(path):
    parts = path.split(os.sep)
    try:
        role_index = parts.index("data") + 1
        return parts[role_index].lower()
    except Exception:
        return "general"  # fallback


def load_documents_from_directory(data_dir):
    documents = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith(".md"):
                loader = UnstructuredMarkdownLoader(full_path)
            elif file.endswith(".csv"):
                loader = CSVLoader(full_path)
            else:
                continue

            try:
                docs = loader.load()
                for doc in docs:
                    doc.metadata["role"] = extract_role_from_path(full_path)
                    doc.metadata["source"] = os.path.basename(full_path)
                    documents.append(doc)
            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")
    return documents


def main():
    # 1. Load .md and .csv documents
    print("📂 Loading documents...")
    documents = load_documents_from_directory(DATA_DIR)

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 3. Deduplicate
    seen = set()
    deduped_docs = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            deduped_docs.append(doc)
    docs = deduped_docs

    # Print metadata for verification
    for i, doc in enumerate(docs):
        print(f"✅ Doc {i} | Role: {doc.metadata['role']} | Source: {doc.metadata['source']}")

    # 4. Manually initialize QdrantClient
    client = QdrantClient(path=QDRANT_PATH)

    if not client.collection_exists(QDRANT_COLLECTION):
        print(f"🛠️  Creating Qdrant collection: {QDRANT_COLLECTION}")
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={"size": 384, "distance": "Cosine"}  # for MiniLM
        )

    # 5. Insert into Qdrant
    vectordb = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embedding_model,
    )

    print("📥 Inserting documents...")
    vectordb.add_documents(docs)
    print("✅ Embeddings successfully stored!")


if __name__ == "__main__":
    main()
