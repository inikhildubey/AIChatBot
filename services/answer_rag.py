import re

import ollama

from services.cleaning_text import clean_text, get_clean_snippet


def build_sources(context_chunks, top_n=3):
    sources = []

    for item in context_chunks[:top_n]:
        text = item['doc']
        score = item['score']
        text = clean_text(text)
        text = re.sub(r'^[^a-zA-Z0-9]+', '', text)
        snippet = get_clean_snippet(text)

        sources.append({
            "snippet": snippet,
            "score": round(score, 2)
        })

    return sources


def generate_answer(query: str, context_chunks: list):
    if not context_chunks:
        return "I don't know."
    context = "\n\n".join(
        f"Chunk {i + 1}: {clean_text(t)}"
        for i, t in enumerate(context_chunks)
    )

    prompt = f"""
    You are a strictly grounded assistant.

    You MUST answer ONLY using the provided context.

    RULES:
    - Use ONLY the given context
    - Do NOT use any external knowledge
    - Do NOT guess or assume missing information
    - Do NOT complete information using prior knowledge

    CRITICAL:
    - If the answer is NOT explicitly present in the context → say:
      "I don't know based on the provided context."

    - If BOTH items (for comparison) are NOT clearly explained → say:
      "I don't know based on the provided context."

    Context:
    {context}

    Question:
    {query}

    Answer ONLY if the context explicitly supports it.
    """
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return response["message"]["content"]
