import io

import numpy as np
import ollama
from fastapi import APIRouter, UploadFile, File
from pypdf import PdfReader


router = APIRouter(prefix="/greet")
embeddings_store = []


@router.get("/")
def greet_welcome(name: str):
    return {"message": f"Welcome to the Greet Module {name}"}




@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    global embeddings_store

    content = await file.read()
    pdf = PdfReader(io.BytesIO(content))

    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    text = text.replace("\n", " ").strip()
    chunks = chunk_text(text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

    embeddings_store = []  # reset

    for idx, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        embeddings_store.append({
            "text": chunk,
            "vector": emb,
            "index": idx
        })
    return {
        "filename": file.filename,
        "chunks": len(chunks)
    }


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def get_embedding(text: str, provider="ollama"):
    if provider == "ollama":
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]


@router.post("/ask")
async def ask_question(query: str):
    global embeddings_store
    if not embeddings_store:
        return {"error": "No document uploaded yet"}
    results, query_vec = search(query, embeddings_store)
    selected_indices = [idx for score, idx in results]

    context_chunks = []

    for idx in selected_indices:
        for neighbor in [idx - 1, idx, idx + 1]:
            if 0 <= neighbor < len(embeddings_store):
                context_chunks.append(embeddings_store[neighbor])
    seen = set()
    unseen_chunks = []
    for item in context_chunks:
        if item['index'] not in seen:
            unseen_chunks.append(item)
            seen.add(item['index'])
    context_chunks = unseen_chunks
    final_chunks = rerank_chunks(context_chunks, query_vec)
    answer = generate_answer(query, final_chunks)

    return {
        "question": query,
        "answer": answer
        # "sources": top_chunks
    }


def search(query, embeddings_store):
    query_vec = get_embedding(query)
    query_lower = query.lower()

    results = []

    for item in embeddings_store:
        text = item["text"]
        text_lower = text.lower()

        # Base semantic score
        score = cosine_similarity(query_vec, item["vector"])

        # Full phrase boost (strong signal)
        if query_lower in text_lower:
            score += 0.5

        # Keyword boost (medium signal)
        keywords = query_lower.split()
        for word in keywords:
            if word in text_lower:
                score += 0.2

        # results.append((score, text))
        results.append((score, item["index"]))
    # Sort by highest score
    results.sort(key=lambda x: x[0], reverse=True)

    # Return top chunks
    return results[:20], query_vec


def rerank_chunks(chunks, query_vec):
    scored = []
    for item in chunks:
        score = cosine_similarity(query_vec, item["vector"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item['text'] for score, item in scored[:5]]  # keep best 5


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def generate_answer(query: str, context_chunks: list):
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant.
                Answer the question ONLY using the context below.
                Use the provided context primarily.
                If partial information is available, answer as completely as possible..
                If the answer is not in the context, say "I don't know".
                Be concise and accurate..

                Context:
                {context}

                Question:
                {query}"""
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]
