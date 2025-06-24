from qdrant_client import QdrantClient

QDRANT_PATH = "/Users/deakshshetty/Documents/RAG-based-role-access/embeddings/qdrant_local"
COLLECTION_NAME = "finrolebot"

client = QdrantClient(path=QDRANT_PATH)

# Scroll through a few points
results, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=20,
    with_payload=True
)

print("\n📦 Sample documents in Qdrant:\n")
for i, point in enumerate(results, 1):
    print(f"🔹 Document {i}")
    print("   ID:", point.id)
    print("   Payload:", point.payload)
    print("--------")
