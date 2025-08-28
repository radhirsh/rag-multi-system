from langchain.embeddings import OllamaEmbeddings

def generate_embeddings(all_chunks: list):
    ollama_model = OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")
    embeddings = []
    for chunk in all_chunks:
        # embed_documents expects a list of texts, returns a list of embeddings
        emb = ollama_model.embed_documents([chunk])
        embeddings.extend(emb)
    return embeddings



