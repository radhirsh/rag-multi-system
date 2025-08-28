from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Split LangChain Document objects into smaller chunks.
    
    Args:
        documents (list): List of LangChain Document objects.
        chunk_size (int): Max chunk size in characters.
        chunk_overlap (int): Overlap between chunks to maintain context.
        
    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        all_chunks.extend(chunks)
        
        
    return all_chunks