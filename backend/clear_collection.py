from qdrant_client import QdrantClient

client = QdrantClient(path="/Users/deakshshetty/Documents/RAG-based-role-access/embeddings/qdrant_local")
client.delete_collection(collection_name="finrolebot")
print("✅ Deleted collection 'finrolebot'")
