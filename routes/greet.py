import ast
import io
import re
import json
import os
import textwrap

import chromadb
import numpy as np
import ollama
from fastapi import APIRouter, UploadFile, File
from pypdf import PdfReader

router = APIRouter(prefix="/greet")
BASE_DIR = os.path.dirname(__file__)
client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))

# client = chromadb.Client(
#     chromadb.Settings(
#         persist_directory="./chroma_db"
#     )
# )

collection = client.get_or_create_collection(name="documents")

embeddings_store = []


# if os.path.exists("embeddings.json"):
#     with open("embeddings.json", "r") as f:
#         embeddings_store = json.load(f)
#     for item in embeddings_store:
#         item["vector"] = np.array(item["vector"])


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

        collection.add(documents=[chunk], embeddings=[emb], ids=[str(idx)], metadatas=[{"chunk_id": idx}])

    return {"filename": file.filename, "chunks": len(chunks)}


def chunk_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    chunk_size = 2  # 2–3 sentences per chunk

    chunks = [
        " ".join(sentences[i:i + chunk_size])
        for i in range(0, len(sentences), chunk_size)
    ]

    return chunks


def get_embedding(text: str, provider="ollama"):
    if provider == "ollama":
        response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        return response["embedding"]


@router.post("/ask")
async def ask_question(query: str):
    query = query.lower().replace('"', '').replace('?', '').strip()
    query = re.sub(r"\s+", " ", query)
    context_chunks = search(query)
    chunk_text = []
    for score, text in context_chunks:
        chunk_text.append(text)

    answer = generate_answer(query, chunk_text)
    return {"question": query, "answer": answer, "chunks": build_sources(context_chunks)}


def build_sources(context_chunks, top_n=3):
    sources = []

    for score, text in context_chunks[:top_n]:
        text = clean_text(text)
        text = re.sub(r'^[^a-zA-Z0-9]+', '', text)
        snippet = get_clean_snippet(text)

        sources.append({
            "snippet": snippet,
            "score": round(score, 2)
        })

    return sources


def get_clean_snippet(text):
    sentences = text.split(".")

    for s in sentences:
        s = s.strip()

        # ignore very short / broken sentences
        if len(s) > 40:
            return textwrap.shorten(s, width=500, placeholder="...")

    # fallback if nothing found
    return textwrap.shorten(text, width=500, placeholder="...")


def search(query, top_k=25):
    queries = [query]

    rewritten = rewrite_query(query)

    for q in rewritten:
        q = q.lower().strip()
        q = re.sub(r"\s+", " ", q)
        if q not in queries:
            queries.append(q)

    # store chunks per query separately
    all_results = []

    for q in queries:
        query_vec = get_embedding(q)

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k
        )

        chunks = top_chunks(results, q)

        # keep top chunks per query
        all_results.append(chunks[:10])

    # 🔥 merge fairly
    context_chunks = merge_round_robin(all_results, top_k=20)
    unique_chunks = []
    seen_docs = set()

    for iteam in context_chunks:
        doc = iteam.get('doc')
        normalized = clean_text(doc)

        if normalized not in seen_docs:
            seen_docs.add(normalized)
            unique_chunks.append(iteam)
    return [(item["score"], item["doc"]) for item in unique_chunks[:20]]


def top_chunks(results, query):
    context_chunks = []
    stop_words = {
        "what", "is", "the", "does", "are", "a", "an", "of",
        "between", "difference", "compare"
    }

    ids = results.get("ids")[0]
    distances = results.get("distances")[0]

    for idx_str, dist in zip(ids, distances):
        idx = int(idx_str)
        vector_score = 1 - dist
        neighbors = [idx]

        if vector_score > 0.7:
            neighbors.extend([idx - 1, idx + 1])

        for neighbor in neighbors:
            if neighbor < 0:
                continue

            data = collection.get(ids=[str(neighbor)])
            doc = data.get("documents")

            if not doc:
                continue

            doc = doc[0].lower()
            score = 0
            match_count = 0

            # 🔥 vector score
            score += vector_score * 2  # slightly stronger

            # 🔥 phrase match
            if query in doc:
                score += 3
                match_count += 1

            # 🔥 acronym boost (MOVE BEFORE FILTER)
            if len(query) <= 5:
                if f"({query})" in doc or doc.startswith(query):
                    score += 2
                    match_count += 1

            # 🔥 word matching
            for word in query.split():
                if word in stop_words:
                    continue

                if word in doc:
                    match_count += 1

                    if len(word) > 4:
                        score += 2
                    else:
                        score += 0.5

            comparison_words = {"difference", "compare", "vs"}
            query_words = set(query.split())
            if query_words.intersection(comparison_words):
                if any(word in doc for word in ["whereas", "while", "difference", "compared"]):
                    score += 3
                    match_count += 1

            # 🔥 smart filter
            if match_count == 0 and vector_score < 0.6:
                continue

            context_chunks.append({
                "score": score,
                "doc": doc,
                "vector_score": round(vector_score, 3),
                "match_count": match_count
            })
    context_chunks.sort(key=lambda x: x["score"], reverse=True)
    return context_chunks


def merge_round_robin(all_results, top_k=20):
    merged = []

    # track current index for each query result list
    pointers = [0] * len(all_results)

    while len(merged) < top_k:
        added_any = False

        for i, chunk_list in enumerate(all_results):

            # if current query still has chunks left
            if pointers[i] < len(chunk_list):

                merged.append(chunk_list[pointers[i]])

                # move pointer forward
                pointers[i] += 1

                added_any = True

                # stop if enough chunks collected
                if len(merged) >= top_k:
                    break

        # if no query has chunks left
        if not added_any:
            break

    return merged


def generate_answer(query: str, context_chunks: list):
    if not context_chunks:
        return "I don't know."
    context = "\n\n".join(
        f"Chunk {i + 1}: {clean_text(t)}"
        for i, t in enumerate(context_chunks)
    )
    prompt = f"""
    You are a grounded assistant.

    You MUST answer using ONLY the provided context.
    RULES:
    - Use ONLY the given context
    - You MAY combine information from multiple chunks
    - You MAY infer simple relationships (like comparison or explanation)
    - Do NOT use external knowledge
    IMPORTANT:
    - If NO relevant information exists → say "I don't know"
    - If information is partially available → answer using what is available

    Context:
    {context}

    Question:
    {query}

    Answer clearly and concisely.
    """
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return response["message"]["content"]


def rewrite_query(query: str):
    prompt = f"""
        You are a search query optimizer.
        
        Rewrite the user query into 2-3 short search queries
        that will help retrieve relevant documents.
        
        Rules:
        - Keep them concise
        - Include key concepts
        - Keep intent (comparison, definition, etc.)
        - Output ONLY Python list
        
        Examples:
        Query: difference between rtgs and neft
        Output: ["rtgs vs neft difference", "rtgs definition", "neft definition"]
        
        Query: what is crr
        Output: ["crr meaning", "cash reserve ratio definition"]
        
        Query: {query}
        """
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    content = response["message"]["content"]

    try:
        return ast.literal_eval(content)
    except:
        return [query]


def clean_text(text):
    text = re.sub(r'[\uf000-\uf0ff]', '', text)  # remove weird unicode bullets
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text.strip()
