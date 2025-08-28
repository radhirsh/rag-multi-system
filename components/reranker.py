import requests

def rerank_with_qwen_ollama(query: str, docs: list[str], top_k: int = 5):
    """
    Rerank documents using Qwen3-Reranker-8B via Ollama API.
    
    Args:
        query (str): The user query.
        docs (list[str]): List of document texts to be reranked.
        top_k (int): Number of top documents to return.
        
    Returns:
        List[str]: Top-k reranked documents.
    """
    scores = []
    for doc in docs:
        prompt = f"Query: {query}\nDocument: {doc}\nScore the relevance of the document to the query on a scale from 1 (not relevant) to 5 (very relevant). Only return the number."
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "Qwen3",
                "prompt": prompt,
                "stream": False
            }
        )
        try:
            score_text = response.json()["response"].strip()
            score = float(score_text.split()[0])  # crude extraction
            scores.append((doc, score))
        except Exception as e:
            print("Error scoring doc:", e)
            scores.append((doc, 0))  # fallback to 0 score
    
    # Sort documents by score descending
    sorted_docs = sorted(scores, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:top_k]]
