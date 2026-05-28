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
    text = clean_text(text)
    cleaned = clean_text(text)
    paragraphs = cleaned.split("\n")
    filtered_text = [p for p in paragraphs if not is_noise(p) and len(p.split()) > 8]

    chunks = chunk_text(filtered_text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

    for idx, chunk in enumerate(chunks):
        print("Chunk Number:", idx, "\nLength of chunk:", len(chunk))
        print("Chunk text:", chunk)
        emb = get_embedding(chunk)

        collection.add(documents=[chunk], embeddings=[emb], ids=[str(idx)], metadatas=[{"chunk_id": idx}])

    return {"filename": file.filename, "chunks": len(chunks)}


# def chunk_text(text, chunk_size=500):
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     chunk_size = 2  # 2–3 sentences per chunk
#
#     chunks = [
#         " ".join(sentences[i:i + chunk_size])
#         for i in range(0, len(sentences), chunk_size)
#     ]
#     chunks = [c.strip() for c in chunks if 80 < len(c) < 500]
#     return chunks


# def chunk_text(text, chunk_size=300):
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#
#     chunks = []
#     current_chunk = ""
#
#     for sent in sentences:
#         if not sent.strip():
#             continue
#
#         if len(current_chunk) + len(sent) < chunk_size:
#             current_chunk += " " + sent
#         else:
#             chunks.append(current_chunk.strip())
#
#             # overlap
#             words = current_chunk.split()
#             overlap_words = words[-20:]
#             current_chunk = " ".join(overlap_words) + " " + sent
#
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#
#     return chunks

# def chunk_text(text, min_words=80, max_words=150):
#     words = text.split()
#     chunks = []
#
#     current_chunk = []
#
#     for word in words:
#         current_chunk.append(word)
#
#         if len(current_chunk) >= max_words:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = []
#
#     # handle leftover
#     if current_chunk:
#         if len(current_chunk) < min_words and chunks:
#             # merge with last chunk
#             chunks[-1] += " " + " ".join(current_chunk)
#         else:
#             chunks.append(" ".join(current_chunk))
#
#     return chunks

def chunk_text(text, max_words=120, overlap=20):
    chunks = []
    current_chunk = []

    for sent in text:
        words = sent.split()

        #  If sentence itself is too big → fallback to word split
        if len(words) > max_words:
            for i in range(0, len(words), max_words - overlap):
                sub_chunk = words[i:i + max_words]
                chunks.append(" ".join(sub_chunk))
            continue

        # normal sentence accumulation
        if len(current_chunk) + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] + words
        else:
            current_chunk.extend(words)
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text.split()) > 15:  # basic quality check
            chunks.append(chunk_text)

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
    for item in context_chunks:
        text = item["doc"]
        chunk_text.append(text)
    answer = generate_answer(query, chunk_text)
    return {"question": query, "answer": answer, "chunks": build_sources(context_chunks)}


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


def get_clean_snippet(text):
    sentences = text.split(".")

    for s in sentences:
        s = s.strip()

        # ignore very short / broken sentences
        if len(s) > 40:
            return textwrap.shorten(s, width=500, placeholder="...")

    # fallback if nothing found
    return textwrap.shorten(text, width=500, placeholder="...")


def search(query, top_k=5):
    queries = [query]

    rewritten = rewrite_query(query)

    for q in rewritten:
        q = q.lower().strip()
        q = re.sub(r"\s+", " ", q)
        if q not in queries:
            queries.append(q)

    all_vector_chunks = []
    all_keyword_chunks = []

    # 🔹 VECTOR RETRIEVAL
    for q in queries:
        query_vec = get_embedding(q)

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k
        )

        chunks = top_chunks(results, q)  # your existing function
        all_vector_chunks.extend(chunks)

    # 🔹 KEYWORD RETRIEVAL
    keyword_chunks = keyword_search(query)
    all_keyword_chunks.extend(keyword_chunks)

    # 🔹 MERGE BOTH
    merged_chunks = []

    # vector chunks (dict format)
    for item in all_vector_chunks:
        merged_chunks.append((item["score"], item["doc"]))

    # keyword chunks (tuple format)
    for item in all_keyword_chunks:
        score = item.get('score') * 2  # simple keyword score
        merged_chunks.append((item.get('score') * 2, item.get('doc')))

    print("\n Vector chunks:\n", all_vector_chunks)
    print("\n Keyword chunks:\n", all_keyword_chunks)
    # 🔹 DEDUPLICATE
    seen = set()
    unique_chunks = []

    for score, doc in merged_chunks:
        normalized = clean_text(doc)

        if normalized not in seen:
            seen.add(normalized)
            unique_chunks.append((score, doc))

    # 🔹 FINAL SORT
    unique_chunks.sort(key=lambda x: x[0], reverse=True)
    reranked = rerank_chunks(query, unique_chunks)
    return reranked[:6]


def keyword_search(query, max_chunks=30):
    stop_words = {
        "what", "is", "the", "does", "are", "a", "an", "of",
        "between", "difference", "compare", "vs", "how", "why"
    }

    query_words = {
        word.lower()
        for word in query.split()
        if word.lower() not in stop_words and len(word) > 2
    }

    results = []

    # ⚠️ This scans your DB (okay for now)
    all_docs = collection.get(include=["documents"])

    for doc, doc_id in zip(all_docs["documents"], all_docs["ids"]):
        text = doc.lower()

        doc_words = set(re.findall(r"\w+", text))
        overlap = sum(1 for w in query_words if w in doc_words)
        if overlap >= 1 and len(text.split()) > 15:
            results.append({
                "score": overlap * 2,
                "doc": doc,
                "source": "keyword"
            })

    # sort by keyword match strength
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:max_chunks]


def top_chunks(results, query):
    context_chunks = []

    stop_words = {
        "what", "is", "the", "does", "are", "a", "an", "of",
        "between", "difference", "compare", "vs", "how", "why"
    }

    # 🔹 extract meaningful query words
    query_words = {
        word.lower()
        for word in query.split()
        if word.lower() not in stop_words and len(word) > 2
    }

    ids = results.get("ids")[0]
    distances = results.get("distances")[0]

    for idx_str, dist in zip(ids, distances):
        idx = int(idx_str)

        vector_score = 1 - dist

        # 🔹 neighbor expansion (controlled)
        neighbors = [idx]
        if vector_score > 0.7:
            neighbors.extend([idx - 1, idx + 1])

        for neighbor in neighbors:
            if neighbor < 0:
                continue

            data = collection.get(ids=[str(neighbor)])
            doc_list = data.get("documents")

            if not doc_list:
                continue

            doc = doc_list[0].lower()

            # 🔹 keyword overlap (CORE PART)
            doc_words = set(doc.split())
            keyword_overlap = len(query_words.intersection(doc_words))

            # 🔹 filter (IMPORTANT)
            if keyword_overlap == 0 and vector_score < 0.65:
                continue

            # 🔹 scoring (clean & simple)
            score = 0
            score += vector_score * 2
            score += keyword_overlap * 2

            # 🔹 phrase boost (optional but useful)
            if query in doc:
                score += 3

            context_chunks.append({
                "score": score,
                "doc": doc,
                "vector_score": round(vector_score, 3),
                "keyword_overlap": keyword_overlap,
                "source": "vector"
            })

    # 🔹 sort final chunks
    context_chunks.sort(key=lambda x: x["score"], reverse=True)

    return context_chunks


# def merge_round_robin(all_results, top_k=20):
#     merged = []
#
#     # track current index for each query result list
#     pointers = [0] * len(all_results)
#
#     while len(merged) < top_k:
#         added_any = False
#
#         for i, chunk_list in enumerate(all_results):
#
#             # if current query still has chunks left
#             if pointers[i] < len(chunk_list):
#
#                 merged.append(chunk_list[pointers[i]])
#
#                 # move pointer forward
#                 pointers[i] += 1
#
#                 added_any = True
#
#                 # stop if enough chunks collected
#                 if len(merged) >= top_k:
#                     break
#
#         # if no query has chunks left
#         if not added_any:
#             break
#
#     return merged


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
    # remove weird unicode
    text = re.sub(r'[\uf000-\uf0ff]', '', text)
    # remove long dotted placeholders
    text = re.sub(r'\.{3,}', ' ', text)
    # remove repeated special chars
    text = re.sub(r'[-_]{3,}', ' ', text)
    # fix broken words like "w slr"
    text = re.sub(r'\b[a-z]\s+(?=[a-z])', '', text)
    # normalize spaces BUT preserve newlines
    text = re.sub(r'[ \t]+', ' ', text)
    # remove page numbers like "123"
    text = re.sub(r'\b\d{1,3}\b', ' ', text)
    # fix broken newlines
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def rerank_chunks(query, chunks):
    reranked = []

    # Limit candidates to reduce cost + noise
    for base_score, doc in chunks[:10]:

        prompt = f"""
        You are a strict scoring system.
        
        Query: {query}
        
        Chunk:
        {doc}
        
        Rules:
        
        - If BOTH concepts + clear difference → 9 or 10
        - If BOTH but no difference → 5
        - If only one concept → 2
        - If irrelevant → 0
        
        CRITICAL:
        - Output ONLY ONE NUMBER
        - Do NOT explain
        - Do NOT write text
        - Do NOT write range (like 9-10)
        - Do NOT write sentences
        
        Valid outputs:
        0
        2
        5
        9
        10
        
        ONLY output the number.
        
        IMPORTANT:
        - You MUST differentiate scores
        - Best chunk → 9–10
        - Medium → 5–7
        - Weak → 1–3
        - Do NOT give same score unless truly equal
    """

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        text = response["message"]["content"].strip()

        # 🔍 Debug raw LLM output
        print("RAW LLM OUTPUT:", text)

        # 🔢 Safe score extraction
        match = re.search(r"\b(10|[0-9])\b", text)

        if match:
            llm_score = int(match.group())
        else:
            print("⚠️ Failed to parse LLM output:", text)
            llm_score = 0

        # 🧠 Combine score (LLM primary, base_score for tie-break)
        combined_score = llm_score + (0.01 * base_score)

        reranked.append({
            "score": combined_score,
            "doc": doc,
            "base_score": base_score,
            "llm_score": llm_score
        })

    # 🔽 Sort by final score
    reranked.sort(key=lambda x: x["score"], reverse=True)

    # 📊 Debug final ranking
    print("\n===== FINAL RERANKED =====")
    for i, r in enumerate(reranked):
        print(f"{i + 1}. Score: {r['score']} (LLM: {r['llm_score']}, Base: {r['base_score']})")
        print(r['doc'][:200])
        print("------")

    return reranked


def is_noise(text):
    text = text.lower()

    if any(x in text for x in [
        "practice questions",
        "session",
        "ppt",
        "exercise",
        "objective"
    ]):
        return True

    if "?" in text:
        return True

    return False
