import re


def chunk_text(text, chunk_size=300):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sent in sentences:
        if not sent.strip():
            continue

        if len(current_chunk) + len(sent) < chunk_size:
            current_chunk += " " + sent
        else:
            chunks.append(current_chunk.strip())

            # overlap
            words = current_chunk.split()
            overlap_words = words[-20:]
            current_chunk = " ".join(overlap_words) + " " + sent

    if current_chunk:
        chunks.append(current_chunk.strip())
    for chunk in chunks:
        if (
                "scheduled commercial banks" in chunk.lower()
                or
                "net demand and time liabilities" in chunk.lower()
        ):
            print("=" * 100)
            print(chunk)
    return chunks


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


# def chunk_text(text, max_words=120, overlap=20):
#     chunks = []
#     current_chunk = []
#
#     for sent in text:
#         words = sent.split()
#
#         #  If sentence itself is too big → fallback to word split
#         if len(words) > max_words:
#             for i in range(0, len(words), max_words - overlap):
#                 sub_chunk = words[i:i + max_words]
#                 chunks.append(" ".join(sub_chunk))
#             continue
#
#         # normal sentence accumulation
#         if len(current_chunk) + len(words) > max_words:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = current_chunk[-overlap:] + words
#         else:
#             current_chunk.extend(words)
#     if current_chunk:
#         chunk_text = " ".join(current_chunk)
#         if len(chunk_text.split()) > 15:  # basic quality check
#             chunks.append(chunk_text)
#
#     return chunks

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
