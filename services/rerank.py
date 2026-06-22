from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# def rerank_chunks(query, chunks):
#     reranked = []
#
#     # Limit candidates to reduce cost + noise
#     for base_score, doc in chunks[:10]:
#
#         prompt = f"""
#         Question:
#         {query}
#
#         Chunk:
#         {doc}
#
#         Can this chunk ALONE answer the question?
#
#         Scoring:
#
#         10 = Direct answer present
#         7 = Most of answer present
#         3 = Related topic only
#         0 = Cannot answer
#
#         Examples:
#
#         Question: What is CRR?
#
#         Chunk: CRR stands for Cash Reserve Ratio.
#         Score: 10
#
#         Chunk: RBI may increase CRR to control inflation.
#         Score: 3
#
#         Chunk: NEFT is a payment system.
#         Score: 0
#
#         Return ONLY the score.
#         """
#         response = ollama.chat(
#             model="llama3",
#             messages=[{"role": "user", "content": prompt}],
#             options={"temperature": 0}
#         )
#         text = response["message"]["content"].strip()
#
#         # 🔢 Safe score extraction
#         match = re.search(r"\b(10|[0-9])\b", text)
#
#         if match:
#             llm_score = int(match.group())
#         else:
#             print("⚠️ Failed to parse LLM output:", text)
#             llm_score = 0
#
#         # 🧠 Combine score (LLM primary, base_score for tie-break)
#         combined_score = llm_score + (0.01 * base_score)
#
#         reranked.append({
#             "score": combined_score,
#             "doc": doc,
#             "base_score": base_score,
#             "llm_score": llm_score
#         })
#
#     # 🔽 Sort by final score
#     reranked.sort(key=lambda x: x["score"], reverse=True)
#
#     # 📊 Debug final ranking
#     print("\n===== FINAL RERANKED =====")
#     for i, r in enumerate(reranked):
#         print(f"{i + 1}. Score: {r['score']} (LLM: {r['llm_score']}, Base: {r['base_score']})")
#         print(r['doc'])
#         print("------")
#
#     return reranked


def rerank_chunks(query, chunks):
    if not chunks:
        return []

    pairs = []

    for score, doc in chunks[:10]:
        pairs.append((query, doc))

    ce_scores = cross_encoder.predict(pairs)

    reranked = []

    for (base_score, doc), ce_score in zip(chunks[:10], ce_scores):
        reranked.append({
            "score": float(ce_score),
            "doc": doc,
            "base_score": base_score,
            "cross_encoder_score": float(ce_score)
        })

    reranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print("\n===== CROSS ENCODER RERANKED =====")

    for i, r in enumerate(reranked):
        print(
            f"{i + 1}. Score: {r['score']:.4f}"
        )

        print(r["doc"][:300])

        print("------")

    return reranked
