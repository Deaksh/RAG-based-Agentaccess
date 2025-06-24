from qdrant_client import QdrantClient

QDRANT_PATH = "/Users/deakshshetty/Documents/RAG-based-role-access/embeddings/qdrant_local"
QDRANT_COLLECTION = "finrolebot"

client = QdrantClient(path=QDRANT_PATH)

results = client.search(
    collection_name=QDRANT_COLLECTION,
    query_vector=[0.0]*384,  # Dummy vector of same dimension as embeddings (adjust dim if needed)
    limit=5
)

for point in results:
    print(f"ID: {point.id}")
    print(f"Payload: {point.payload}")
    print("-----")
