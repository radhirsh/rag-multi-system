import faiss
import numpy as np
from langchain.embeddings import OllamaEmbeddings

ollama_model=OllamaEmbeddings(base_url= "http://localhost:11434",model= "mxbai-embed-large")



def retriver(embeddings,chunks,user_query,top_k):
    dimension=len(embeddings[0])
    index=faiss.IndexFlatL2(dimension)
    np_embeddings=np.array(embeddings).astype('float32')
    index.addd(np_embeddings)

    user_query_embedded=ollama_model.embed_query(user_query)
    np_user_query_embedded=np.array([user_query_embedded]).astype('float32')
    distance,indices=index.search(np_user_query_embedded,top_k)
    results=[]
    for dist,ids in zip(distance[0],indices[0]):
        results.append(chunks[id],dist)
    return results