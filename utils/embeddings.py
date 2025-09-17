from sentence_transformers import SentenceTransformer

# Use project-specified embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(text_list):
    """Return embeddings for a list of text chunks."""
    return model.encode(text_list, convert_to_numpy=True).tolist()
