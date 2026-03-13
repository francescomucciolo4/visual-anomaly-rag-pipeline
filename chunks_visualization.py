import chromadb

client = chromadb.PersistentClient(path="chroma_db")

# Vedi le collezioni disponibili
collections = client.list_collections()
print(collections)

# Apri la collezione
collection = client.get_collection("langchain")

# Vedi tutti i chunk
results = collection.get(include=["documents", "embeddings"])

for i, doc in enumerate(results["documents"]):
    print(f"\n--- Chunk {i+1} ---")
    print(doc)
    print(f"Embedding (prime 5 dims): {results['embeddings'][i][:5]}")