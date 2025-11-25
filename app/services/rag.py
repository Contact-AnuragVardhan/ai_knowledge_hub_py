#app/services/rag.py

from typing import List, Tuple, Literal
import re

from openai import OpenAI

from .local_embeddings import embed_text
from .vector_store import VectorStore
from app.config import settings
from app.utils.logging import logger

client = OpenAI(api_key=settings.openai_api_key)

CHAT_MODEL = "gpt-4o-mini"

QueryType = Literal["generic", "specific"]

GENERIC_PATTERNS = [
    r"\bwhat is this (doc|document|file) about\b",
    r"\bsummar(y|ise|ize) (this|the) (doc|document|file)?\b",
    r"\bsummar(y|ise|ize)\b",
    r"\bgive me (an )?overview\b",
    r"\bhigh[- ]level (view|summary)\b",
    r"\bkey points\b",
    r"\bmain (idea|points)\b",
    r"\btl;dr\b",
    r"\bexplain (this|the) (doc|document|file)?\b",
]

STOPWORDS = {
    "what",
    "is",
    "this",
    "the",
    "a",
    "an",
    "of",
    "about",
    "for",
    "in",
    "on",
    "to",
    "me",
    "give",
    "you",
    "please",
    "can",
    "could",
    "would",
    "explain",
    "tell",
    "show",
    "summarize",
    "summarise",
    "summary",
    "overview",
    "document",
    "doc",
    "file",
}


def classify_query(query: str) -> QueryType:
    """
    Heuristic:
    - If it matches 'summarize/what is this document about' patterns -> generic
    - If very short and mostly stopwords -> generic
    - Else -> specific
    """
    q = (query or "").lower().strip()
    if not q:
        return "generic"

    # 1. Pattern-based detection
    for pattern in GENERIC_PATTERNS:
        if re.search(pattern, q):
            logger.debug(f"classify_query: matched generic pattern '{pattern}'")
            return "generic"

    # 2. Token / stopword heuristic
    tokens = re.findall(r"\w+", q)
    if not tokens:
        return "generic"

    stop_count = sum(1 for t in tokens if t in STOPWORDS)
    stop_ratio = stop_count / len(tokens)

    if len(tokens) <= 5 and stop_ratio >= 0.6:
        logger.debug(
            f"classify_query: short+stopword-heavy (len={len(tokens)}, "
            f"stop_ratio={stop_ratio:.2f}) -> generic"
        )
        return "generic"

    logger.debug(
        f"classify_query: treating as specific (len={len(tokens)}, "
        f"stop_ratio={stop_ratio:.2f})"
    )
    return "specific"


def merge_results(
    bm25_rows,
    vec_rows,
    k: int,
):
    """
    Merge BM25 and vector rows, dedupe by (doc_name, chunk_index).
    Works with SQLAlchemy Row objects.
    """
    seen = set()
    merged = []

    for row in list(bm25_rows) + list(vec_rows):
        # We now select chunk_index in both queries
        doc_name = getattr(row, "doc_name", None)
        chunk_index = getattr(row, "chunk_index", None)
        key = (doc_name, chunk_index)

        if key in seen:
            continue

        seen.add(key)
        merged.append(row)

        if len(merged) >= k:
            break

    return merged


def build_context_and_sources(rows, max_chars: int = 12_000):
    """
    Build context string and sources list from merged rows.
    Assumes columns: doc_name, chunk_index, content.
    """
    parts = []
    sources: List[str] = []
    total = 0

    for row in rows:
        content = getattr(row, "content", "") or ""
        if not content:
            continue

        if total + len(content) > max_chars and parts:
            break

        doc_name = getattr(row, "doc_name", "unknown")
        chunk_index = getattr(row, "chunk_index", None)

        parts.append(
            f"[doc={doc_name}, chunk={chunk_index}] {content}"
        )
        sources.append(f"{doc_name}#chunk-{chunk_index}")
        total += len(content)

    context = "\n\n".join(parts)
    return context, sources


def call_chat_model(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def fetch_doc_chunks_for_summary(
    store: VectorStore,
    user_id: int,
    doc_name: str,
    max_chars: int = 12_000,
) -> List[str]:
    """
    Get chunks for a single doc in order and cap by total characters.
    Uses VectorStore.get_chunks_for_doc().
    """
    rows = store.get_chunks_for_doc(user_id, doc_name)

    chunks: List[str] = []
    total = 0

    for row in rows:
        content = getattr(row, "content", "") or ""
        if not content:
            continue

        if total + len(content) > max_chars and chunks:
            break

        chunks.append(content)
        total += len(content)

    return chunks


def summarize_document(
    store: VectorStore,
    user_id: int,
    doc_name: str,
    query: str,
) -> Tuple[str, List[str]]:
    """
    Summarize a single document when the query is generic,
    e.g. "what is this document about".
    """
    logger.info(
        f"summarize_document: user_id={user_id}, doc_name={doc_name}, "
        f"query='{query}'"
    )

    chunks = fetch_doc_chunks_for_summary(store, user_id, doc_name)
    if not chunks:
        logger.warning(
            f"summarize_document: no chunks found for doc_name={doc_name}, "
            f"user_id={user_id}"
        )
        return "I could not find any content for this document.", []

    doc_text = "\n\n".join(chunks)

    system_prompt = (
        "You are an AI assistant that summarizes documents for the user. "
        "You are given text that all comes from a single document."
    )

    user_prompt = (
        f"The user has selected a single document named '{doc_name}'. "
        f"Here is the content (possibly truncated):\n\n"
        f"{doc_text}\n\n"
        f"The user asked: '{query}'. If the question is generic, "
        f"give a clear, concise summary of what this document is about in 3–5 bullet points. "
        f"If the question is more specific, still answer it using only this document."
    )

    answer = call_chat_model(system_prompt, user_prompt)
    sources = [f"{doc_name}#summary"]
    return answer, sources


def answer_query(
    store: VectorStore,
    user_id: int,
    doc_name: str | None,
    query: str,
) -> tuple[str, List[str]]:
    """
    Hybrid RAG:
    - If doc_name is provided and query is 'generic' (what is this doc about, summarize this),
      skip retrieval and summarize that document.
    - Otherwise:
      - BM25 keyword search using Postgres full-text (content_tsv)
      - Semantic search using pgvector
      - Merge + dedupe results, then send to GPT-4o-mini
    """
    logger.info(
        f"RAG answer_query called: user_id={user_id}, doc_name={doc_name}, "
        f"query='{query[:100]}{'...' if len(query) > 100 else ''}'"
    )

    # 1) If we have a selected doc, and the query is generic -> summarization path
    if doc_name:
        qtype = classify_query(query)
        logger.info(
            f"Query classification for user_id={user_id}, doc_name={doc_name}: {qtype}"
        )
        if qtype == "generic":
            return summarize_document(store, user_id, doc_name, query)

    # 2) Standard hybrid RAG path

    # 2.1 Embed query for semantic search
    q_vec = embed_text(query)
    if not q_vec:
        logger.warning("answer_query: empty embedding for query; vector search skipped")

    # For now, use the same top_k setting for both modes
    k = settings.top_k

    # 2.2 Retrieve candidates
    bm25_rows = store.search_bm25(user_id, query, k, doc_name)
    vec_rows = store.top_k(user_id, q_vec, k, doc_name) if q_vec else []

    logger.info(
        f"Hybrid retrieval: BM25={len(bm25_rows)} rows, semantic={len(vec_rows)} rows"
    )

    if not bm25_rows and not vec_rows:
        logger.warning("No retrieval results; returning fallback answer")
        return (
            "I could not find any relevant content for your question in your documents.",
            [],
        )

    # 2.3 Merge & dedupe
    combined_rows = merge_results(bm25_rows, vec_rows, k)
    logger.info(f"Hybrid merged rows count: {len(combined_rows)}")

    # 2.4 Build context
    context, sources = build_context_and_sources(combined_rows)

    system_prompt = (
        "You are an assistant that answers using only the given context. "
        "If the answer is not in the context, say you don't know."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer using only the context above."
    )

    try:
        answer = call_chat_model(system_prompt, user_prompt)
        logger.info(
            f"RAG answer generated successfully: answer_len={len(answer)}, "
            f"sources={len(sources)}"
        )
        return answer, sources
    except Exception as exc:
        logger.exception(f"Chat completion failed: {exc}")
        raise
