from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
import os



def load_documents(filename,path):
    """
    Load documents of various types (txt, pdf, csv) from data_dir.
    Returns a list of LangChain Document objects.
    """
    documents = []
    if filename.endswith(".txt"):
            loader = TextLoader(path)
            docs = loader.load()
            documents.extend(docs)
    elif filename.endswith(".docx"):
        loader=UnstructuredWordDocumentLoader(path)
        docs=loader.load()
        documents.extend(docs)
    elif filename.endswith(".pdf"):
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
    elif filename.endswith(".csv"):
        loader = CSVLoader(path)
        docs = loader.load()
        documents.extend(docs)
    else:
        print(f"Unsupported file type: {filename}")

    
        

    return documents