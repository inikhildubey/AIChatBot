import ast
import re

import ollama

from database.bm25_store import bm25_store
from database.chroma_client import get_collection
from services.cleaning_text import clean_text
from services.embeddings import get_embedding
from services.rerank import rerank_chunks


# embeddings_store = []
# if os.path.exists("embeddings.json"):
#     with open("embeddings.json", "r") as f:
#         embeddings_store = json.load(f)
#     for item in embeddings_store:
#         item["vector"] = np.array(item["vector"])


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
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}], options={"temperature": 0})

    content = response["message"]["content"]

    try:
        return ast.literal_eval(content)
    except:
        return [query]


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

        results = get_collection().query(query_embeddings=[query_vec], n_results=top_k)

        chunks = top_chunks(results, q)  # your existing function
        all_vector_chunks.extend(chunks)
    # 🔹 KEYWORD RETRIEVAL
    # keyword_chunks = keyword_search(query)
    keyword_chunks = bm25_search(query)
    all_keyword_chunks.extend(keyword_chunks)
    rrf_chunks = reciprocal_rank_fusion(all_vector_chunks, all_keyword_chunks)

    # # 🔹 MERGE BOTH
    # merged_chunks = []
    #
    # # Vector chunks
    # for item in all_vector_chunks:
    #     merged_chunks.append((
    #         1,  # neutral base score
    #         item["doc"]
    #     ))
    #
    # # BM25 chunks
    # for item in all_keyword_chunks:
    #     merged_chunks.append((
    #         1,  # neutral base score
    #         item["doc"]
    #     ))

    print("\n Vector chunks:\n", all_vector_chunks)
    print("\n Keyword chunks:\n", all_keyword_chunks)
    print("\n===== RRF RESULTS =====")

    for score, doc in rrf_chunks[:10]:
        print(score)
        print(doc[:150])
        print("------")
    # 🔹 DEDUPLICATE
    seen = set()
    unique_chunks = []
    for score, doc in rrf_chunks:
        normalized = clean_text(doc)
        if normalized not in seen:
            seen.add(normalized)
            unique_chunks.append((score, doc))
    # 🔹 FINAL SORT
    unique_chunks.sort(key=lambda x: x[0], reverse=True)
    reranked = rerank_chunks(query, unique_chunks)
    return reranked[:6]


def top_chunks(results, query):
    context_chunks = []

    stop_words = {"what", "is", "the", "does", "are", "a", "an", "of", "between", "difference", "compare", "vs", "how",
                  "why"}

    # 🔹 extract meaningful query words
    query_words = {word.lower() for word in query.split() if word.lower() not in stop_words and len(word) > 2}

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

            data = get_collection().get(ids=[str(neighbor)])
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

            context_chunks.append(
                {"score": score, "doc": doc, "vector_score": round(vector_score, 3), "keyword_overlap": keyword_overlap,
                 "source": "vector"})

    # 🔹 sort final chunks
    context_chunks.sort(key=lambda x: x["score"], reverse=True)

    return context_chunks


# def keyword_search(query, max_chunks=30):
#     stop_words = {
#         "what", "is", "the", "does", "are", "a", "an", "of",
#         "between", "difference", "compare", "vs", "how", "why"
#     }
#
#     query_words = {
#         word.lower()
#         for word in query.split()
#         if word.lower() not in stop_words and len(word) > 2
#     }
#
#     results = []
#
#     # This scans your DB (okay for now)
#     all_docs = collection.get(include=["documents"])
#
#     for doc, doc_id in zip(all_docs["documents"], all_docs["ids"]):
#         text = doc.lower()
#
#         doc_words = set(re.findall(r"\w+", text))
#         overlap = sum(1 for w in query_words if w in doc_words)
#         if overlap >= 1 and len(text.split()) > 15:
#             results.append({
#                 "score": overlap * 2,
#                 "doc": doc,
#                 "source": "keyword"
#             })
#
#     # sort by keyword match strength
#     results.sort(key=lambda x: x["score"], reverse=True)
#
#     return results[:max_chunks]

def bm25_search(query, top_k=5):
    bm25 = bm25_store["documents"]["bm25"]
    chunks = bm25_store["documents"]["chunks"]

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    results = []

    for score, doc in ranked[:top_k]:
        results.append({"score": round(score, 3), "doc": doc, "source": "bm25"})

    return results


def reciprocal_rank_fusion(vector_chunks, bm25_chunks, k=60):
    scores = {}

    # Vector Results
    for rank, item in enumerate(vector_chunks, start=1):

        doc = item["doc"]

        if doc not in scores:
            scores[doc] = 0

        scores[doc] += 1 / (k + rank)

    # BM25 Results
    for rank, item in enumerate(bm25_chunks, start=1):

        doc = item["doc"]

        if doc not in scores:
            scores[doc] = 0

        scores[doc] += 1 / (k + rank)

    # Sort by RRF score
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    final_chunks = []

    for doc, score in sorted_results:
        final_chunks.append((score, doc))

    return final_chunks

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
