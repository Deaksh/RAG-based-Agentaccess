from qdrant_client import QdrantClient

QDRANT_PATH = "/Users/deakshshetty/Documents/RAG-based-role-access/embeddings/qdrant_local"
QDRANT_COLLECTION = "finrolebot"

client = QdrantClient(path=QDRANT_PATH)
schema = client.get_collection(QDRANT_COLLECTION)
print(schema.payload_schema)
