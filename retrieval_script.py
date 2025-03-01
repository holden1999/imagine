import chromadb


# Or chromadb.PersistentClient(path="chroma_db") for persistence
chroma_client = chromadb.HttpClient()

collection = chroma_client.get_collection(
    'media_vectors')  # New collection name for rich features
print(chroma_client.list_collections())
result = collection.query(query_texts=["*"], n_results=2)
print(result)
