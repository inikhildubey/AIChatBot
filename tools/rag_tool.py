import io
import re

from fastapi import UploadFile, File
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from database.bm25_store import bm25_store, save_chunks
from database.chroma_client import client
from services.answer_rag import generate_answer, build_sources
from services.chunking import chunk_text
from services.cleaning_text import clean_text
from services.embeddings import get_embedding
from services.search import search


# if os.path.exists("embeddings.json"):
#     with open("embeddings.json", "r") as f:
#         embeddings_store = json.load(f)
#     for item in embeddings_store:
#         item["vector"] = np.array(item["vector"])


async def upload_data(file: UploadFile = File(...)):

    global collection

    client.delete_collection("documents")
    collection = client.get_or_create_collection("documents")

    content = await file.read()
    pdf = PdfReader(io.BytesIO(content))
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    cleaned = clean_text(text)
    # paragraphs = cleaned.split("\n")
    # filtered_text = [p for p in paragraphs if not is_noise(p) and len(p.split()) > 8]

    # sentences = re.split(r'(?<=[.!?])\s+', cleaned)

    # filtered_text = [s.strip() for s in sentences if len(s.split()) > 5]

    # chunks = chunk_text(filtered_text)
    chunks = chunk_text(cleaned)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]
    # BM25 Indexing
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]

    bm25 = BM25Okapi(tokenized_chunks)

    bm25_store["documents"] = {
        "bm25": bm25,
        "chunks": chunks
    }
    save_chunks(chunks)
    for idx, chunk in enumerate(chunks):
        emb = get_embedding(chunk)

        # collection.add(documents=[chunk], embeddings=[emb], ids=[str(idx)], metadatas=[{"chunk_id": idx}])
        collection.add(
            ids=[str(idx)],
            embeddings=[emb],
            documents=[chunk],
            metadatas=[{
                "chunk_id": idx,
                "source_file": file.filename
            }]
        )

    print("Total chunks:", len(chunks))
    sizes = [len(c.split()) for c in chunks]
    print("Min:", min(sizes))
    print("Max:", max(sizes))
    print("Avg:", sum(sizes) / len(sizes))

    return {"filename": file.filename, "chunks": len(chunks)}


async def ask_question(decision: dict):
    query = decision['query']
    query = query.lower().replace('"', '').replace('?', '').strip()
    query = re.sub(r"\s+", " ", query)
    context_chunks = search(query)
    chunk_text = []
    for item in context_chunks:
        text = item["doc"]
        chunk_text.append(text)
    answer = generate_answer(query, chunk_text)
    return {"question": query, "answer": answer, "chunks": build_sources(context_chunks)}
