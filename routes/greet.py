import io
import json
import os
import re

import chromadb
import numpy as np
import ollama
from fastapi import APIRouter, UploadFile, File
from pypdf import PdfReader

BASE_DIR = os.path.dirname(__file__)

client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))

# client = chromadb.Client(
#     chromadb.Settings(
#         persist_directory="./chroma_db"
#     )
# )

collection = client.get_or_create_collection(name="documents")

router = APIRouter(prefix="/greet")
embeddings_store = []

if os.path.exists("embeddings.json"):
    with open("embeddings.json", "r") as f:
        embeddings_store = json.load(f)
    for item in embeddings_store:
        item["vector"] = np.array(item["vector"])


@router.get("/")
def greet_welcome(name: str):
    return {"message": f"Welcome to the Greet Module {name}"}


@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    global collection

    client.delete_collection("documents")
    collection = client.get_or_create_collection("documents")

    content = await file.read()
    pdf = PdfReader(io.BytesIO(content))

    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    text = text.replace("\n", " ").strip()
    chunks = chunk_text(text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

    for idx, chunk in enumerate(chunks):
        emb = get_embedding(chunk)

        collection.add(documents=[chunk], embeddings=[emb], ids=[str(idx)])

    return {"filename": file.filename, "chunks": len(chunks)}


def chunk_text(text, chunk_size=500, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def get_embedding(text: str, provider="ollama"):
    if provider == "ollama":
        response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        return response["embedding"]


@router.post("/ask")
async def ask_question(query: str):
    query = query.lower().replace('"', '').replace('?', '').strip()
    context_chunks = search(query)
    final_chunks = []
    for score, text in context_chunks:
        final_chunks.append(text)

    answer = generate_answer(query, final_chunks)

    return {"question": query, "answer": answer}


def search(query, top_k=25):
    query_vec = get_embedding(query)
    queries = decompose_query(query)
    context_chunks = []
    if type(queries) is list:
        for q in queries:
            query_vec = get_embedding(q)

            results = collection.query(query_embeddings=[query_vec], n_results=top_k)
            all_chunks = top_chunks(results, query)
            context_chunks.extend(all_chunks)

    else:
        #
        results = collection.query(query_embeddings=[query_vec], n_results=top_k)
        context_chunks = top_chunks(results, query)
    context_chunks.sort(key=lambda x: x[0], reverse=True)
    return context_chunks[:20]


def top_chunks(results, query):
    context_chunks = []
    seen = set()
    for idx_str in results.get("ids")[0]:
        idx = int(idx_str)
        stop_words = {"what", "is", "the", "does", "are", "a", "an", "of"}
        for neighbor in [idx - 1, idx, idx + 1]:
            score = 0
            if neighbor >= 0:
                data = collection.get(ids=[str(neighbor)])
                doc = data.get("documents")
                if doc and doc[0] not in seen:
                    doc = doc[0].lower()
                    if query in doc:
                        score = score + 1
                    for word in query.split():
                        if word in stop_words:
                            continue
                        if word in doc:
                            score = score + 0.5
                    seen.add(doc)
                    context_chunks.append((score, doc))
    context_chunks.sort(key=lambda x: x[0], reverse=True)
    return context_chunks


def generate_answer(query: str, context_chunks: list):
    # context = "\n\n".join(clean_text(t) for t in context_chunks)
    context = "\n\n".join(f"Chunk {i + 1}: {clean_text(t)}" for i, t in enumerate(context_chunks))

    prompt = f"""You are a helpful assistant.

            Answer the question using ONLY the context below.
            If the question asks for comparison or difference:
            - Identify relevant parts from the context
            - Compare them clearly in bullet points
            Do NOT say "I don't know" if relevant information is present in context.
            Only say "I don't know" if NOTHING relevant exists.
            Context:
            {context}
            
            Question:
            {query}"""
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
    return response["message"]["content"]


def decompose_query(query: str):
    prompt = f"""
You are a query understanding system.

Break the user query into smaller independent search queries.

Rules:
- If it is a comparison, split into separate entities
- If it is a single concept, return only one query
- Keep queries short and precise
- Return ONLY a Python list

Query: {query}
"""

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}], options={"temperature": 0})

    content = response["message"]["content"]

    try:
        return eval(content)  # quick parsing (safe enough here)
    except:
        return [query]


def clean_text(text):
    text = re.sub(r'[\uf000-\uf0ff]', '', text)  # remove weird unicode bullets
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text.strip()
