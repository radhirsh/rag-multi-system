def build_prompt(query: str, docs: list[str]) -> str:
    """
    Builds a prompt to send to the LLM.
    
    Args:
        query (str): The user's question.
        docs (list[str]): Retrieved and reranked context documents.

    Returns:
        str: Final prompt.
    """
    context = "\n\n".join(docs)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:"""
    return prompt
