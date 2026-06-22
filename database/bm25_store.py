# database/bm25_store.py

import json
import os

from rank_bm25 import BM25Okapi

bm25_store = {}

DATA_FILE = "data/chunks.json"


def save_chunks(chunks):
    with open(DATA_FILE, "w") as f:
        json.dump(chunks, f)


def load_bm25():

    if "documents" in bm25_store:
        return
    if not os.path.exists(DATA_FILE):
        return

    with open(DATA_FILE, "r") as f:
        chunks = json.load(f)

    tokenized_chunks = [
        chunk.lower().split()
        for chunk in chunks
    ]

    bm25_store["documents"] = {
        "bm25": BM25Okapi(tokenized_chunks),
        "chunks": chunks
    }