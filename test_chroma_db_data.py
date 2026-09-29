import chromadb

client=chromadb.PersistentClient(path="chroma_db")

collection=client.get_collection(name="documents")

result=collection.query(query_texts=["Contract No: |Contract No: GEMC-511687721084070"],
    n_results=20)
print(result)
