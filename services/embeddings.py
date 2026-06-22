import ollama


def get_embedding(text: str, provider="ollama"):
    if provider == "ollama":
        response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        return response["embedding"]
