import chromadb
from chromadb.config import Settings
from utils.embeddings import get_embeddings

client = chromadb.Client(Settings())

# Multiple collections for query routing
invoice_store = client.get_or_create_collection("invoice_vector_store")
paper_store = client.get_or_create_collection("paper_vector_store")
combined_store = client.get_or_create_collection("combined_store")

def store_chunks(chunks, metadata_list, embeddings, doc_type="manual"):
    """Store chunks in the correct vector store based on doc_type."""
    collection_map = {
        "invoice": invoice_store,
        "research": paper_store,
        "manual": combined_store
    }
    store = collection_map.get(doc_type, combined_store)

    for i, chunk in enumerate(chunks):
        store.add(
            documents=[chunk],
            metadatas=[metadata_list[i]],
            embeddings=[embeddings[i]],
            ids=[str(i)]
        )

def query_chunks(query_text, top_k=5):
    """
    Query routing based on keywords:
    - 'invoice' -> invoice store
    - 'research' -> paper store
    - else -> combined store
    """
    query_lower = query_text.lower()
    if "invoice" in query_lower:
        store = invoice_store
    elif "research" in query_lower:
        store = paper_store
    else:
        store = combined_store

    query_emb = get_embeddings([query_text])[0]

    results = store.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    return results["documents"][0] if results["documents"] else []
