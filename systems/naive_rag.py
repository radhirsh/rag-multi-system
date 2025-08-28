from components.document_loader import load_documents
from components.chunker import chunk_documents
from components.vector_db import vector_db_query

def run_naive_rag(file_path: str, query: str):
    """
    Run a basic RAG pipeline:
    - Load documents
    - Chunk
    - Build or load vector DB
    - Run query against it using LLM
    
    Args:
        file_path (str): Path to input file (PDF, TXT, DOCX, CSV)
        query (str): User's question

    Returns:
        str: Answer from the LLM
    """
    # Load
    filename = file_path.split("\\")[-1]  # or use os.path.basename()
    documents = load_documents(filename, file_path)
    
    # Chunk
    chunks = chunk_documents(documents)

    
    # VectorDB + RetrievalQA
    result = vector_db_query(chunks, query)
    
    return result
