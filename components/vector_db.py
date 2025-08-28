from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import os

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
