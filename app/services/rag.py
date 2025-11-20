from typing import List
from openai import OpenAI

from .embeddings import embed_text
from .vector_store import VectorStore
from app.config import settings
from app.utils.logging import logger

client = OpenAI(api_key=settings.openai_api_key)


def answer_query(
    store: VectorStore,
    user_id: int,
    doc_name: str | None,
    query: str,
) -> tuple[str, List[str]]:
    """
    Hybrid RAG:
    - BM25 keyword search using Postgres full-text (content_tsv)
    - Semantic search using pgvector
    - Merge + dedupe results, then send to GPT-4o-mini
    """
    logger.info(
        f"RAG answer_query called: user_id={user_id}, doc_name={doc_name}, "
        f"query='{query[:100]}{'...' if len(query) > 100 else ''}'"
    )

    # ---- 1. Embed query for semantic search ----
    q_vec = embed_text(query)

    # For now, use the same top_k setting for both modes
    k = settings.top_k

    # ---- 2. Retrieve candidates ----
    bm25_rows = store.search_bm25(user_id, query, k, doc_name)
    vec_rows = store.top_k(user_id, q_vec, k, doc_name)

    logger.info(
        f"Hybrid retrieval: BM25={len(bm25_rows)} rows, semantic={len(vec_rows)} rows"
    )

    # ---- 3. Merge & dedupe (BM25 first, then semantic) ----
    combined_rows = []
    seen = set()  # (doc_name, content) to avoid duplicates

    for row in bm25_rows + vec_rows:
        # row has .doc_name and .content
        key = (row.doc_name, row.content)
        if key in seen:
            continue
        seen.add(key)
        combined_rows.append(row)
        if len(combined_rows) >= k:
            break

    logger.info(f"Hybrid merged rows count: {len(combined_rows)}")

    # ---- 4. Build context for RAG ----
    context = "\n".join(r.content for r in combined_rows)

    prompt = (
        "You are an assistant that answers using only the given context.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.choices[0].message.content
        sources = [r.doc_name for r in combined_rows]

        logger.info(
            f"RAG answer generated successfully: answer_len={len(answer)}, "
            f"sources={len(sources)}"
        )
        return answer, sources
    except Exception as exc:
        logger.exception(f"Chat completion failed: {exc}")
        raise
