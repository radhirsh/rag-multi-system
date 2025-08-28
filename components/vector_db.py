from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from langchain_ollama import OllamaLLM
import os



def store_chunks_in_vector_db(chunks: list[Document], persist_dir: str = "Vector_DB"):
    """
    Stores the document chunks in a Chroma vector database using Ollama embeddings.

    Args:
        chunks (list[Document]): List of LangChain Document chunks.
        persist_dir (str): Directory to persist the Chroma DB.
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")

    # Create Chroma DB from documents and persist it
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print(f"[✔] Chunks stored in vector DB at: {persist_dir}")

def load_vector_db(persist_dir: str = "Vector_DB", embedding_model: str = "mxbai-embed-large"):
    """
    Loads an existing Chroma vector database.

    Args:
        persist_dir (str): Directory where the DB is persisted.
        embedding_model (str): Name of the Ollama embedding model to use.

    Returns:
        Chroma vector DB instance ready for retrieval.
    """
    embeddings = OllamaEmbeddings(model=embedding_model, base_url="http://localhost:11434")

    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"No vector DB found at: {persist_dir}")

    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    print(f"[✔] Vector DB loaded from: {persist_dir}")
    return vectordb


def vector_db_query(docs, user_query, persist_dir="Vector_DB"):
    """
    Build/load vector DB from docs and query it with user_query.
    
    Args:
        docs (List[Document]): List of LangChain Document objects.
        user_query (str): The query string.
        persist_dir (str): Directory to store vector DB.
    
    Returns:
        str: Answer from the retrieval QA chain.
    """
    llm = OllamaLLM(model="llama3.2")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        # Build and persist vector DB if doesn't exist
        vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
        vectordb.persist()
    else:
        # Load existing vector DB
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    result = qa_chain.run(user_query)
    return result
