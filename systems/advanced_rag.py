from components.document_loader import load_documents
from components.chunker import chunk_documents
from components.vector_db import vector_db_query  # If needed for storing chunks
from langchain_ollama import OllamaLLM
from components.reranker import rerank_with_qwen_ollama  # Your custom reranker
from components.vector_db import store_chunks_in_vector_db, load_vector_db
from langchain_community.embeddings import OllamaEmbeddings

def run_advanced_rag(file_path: str, query: str, persist_dir="Vector_DB"):
    """
    Advanced RAG:
    - Chunk + Embed + Persist
    - Retrieve top-k from Vector DB
    - Rerank using Qwen3-Reranker-8B (via Ollama)
    - Answer using llama3.2 (via Ollama)
    """
    # Step 1: Load and chunk
    filename = file_path.split("\\")[-1]
    documents = load_documents(filename, file_path)
    chunks = chunk_documents(documents)

    # Step 2: Store chunks in vector DB (if not already)
    store_chunks_in_vector_db(chunks, persist_dir=persist_dir)

    # Step 3: Load vector DB & retrieve top-k
    vectordb = load_vector_db(persist_dir, embedding_model="mxbai-embed-large")
    retriever = vectordb.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(query)

    # Step 4: Rerank using Qwen3-Reranker-8B
    chunk_texts = [doc.page_content for doc in retrieved_docs]
    reranked_chunks = rerank_with_qwen_ollama(query, chunk_texts, top_k=5)

    # Step 5: Pass reranked context to LLM (llama3.2)
    context = "\n\n".join(reranked_chunks)
    llm = OllamaLLM(model="llama3.2")

    prompt = f"""Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"""
    response = llm.invoke(prompt)

    return response
