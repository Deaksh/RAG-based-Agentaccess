# test_dump_qdrant.py
from qdrant_client import QdrantClient

QDRANT_PATH = "/Users/deakshshetty/Documents/RAG-based-role-access/embeddings/qdrant_local"
COLLECTION_NAME = "finrolebot"

client = QdrantClient(path=QDRANT_PATH)

print("📥 Dumping all points...\n")
points = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=100
)

for point in points[0]:
    payload = point.payload
    meta = payload.get('metadata', {})
    print(f"✅ ID: {point.id}")
    print(f"🔹 Role: {meta.get('role')}")
    print(f"🔹 Source: {meta.get('source')}")
    print("—" * 30)
