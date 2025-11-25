#app/services/vector_store.py

from sqlalchemy import text
from sqlalchemy.orm import Session
from app.utils.logging import logger


class VectorStore:
    def __init__(self, db: Session):
        self.db = db
        logger.debug("VectorStore instance created")

    def insert_chunk(self, user_id: int, doc_name: str, index: int, content: str, embedding):
        logger.info(
            f"Inserting chunk into document_chunks: "
            f"user_id={user_id}, doc_name={doc_name}, index={index}, content_len={len(content)}"
        )
        try:
            # content_tsv is a GENERATED column in Postgres, computed from 'content'
            self.db.execute(
                text(
                    """
                    INSERT INTO document_chunks (user_id, doc_name, chunk_index, content, embedding)
                    VALUES (:user_id, :doc_name, :idx, :content, :embedding)
                    """
                ),
                {
                    "user_id": user_id,
                    "doc_name": doc_name,
                    "idx": index,
                    "content": content,
                    "embedding": embedding,
                },
            )
            logger.debug("Chunk insertion committed (pending outer commit)")
        except Exception as exc:
            self.db.rollback()
            logger.exception(f"Error inserting chunk for {doc_name}, index={index}: {exc}")
            raise

    def top_k(self, user_id: int, query_vec, k: int, doc_name: str | None):
        """
        Semantic search using pgvector (cosine distance).
        Now also selects chunk_index for better dedup + source tracking.
        """
        logger.info(
            f"Querying top_k={k} chunks from document_chunks: user_id={user_id}, doc_name={doc_name}"
        )
        try:
            if doc_name is None:
                sql = """
                    SELECT
                        doc_name,
                        chunk_index,
                        content,
                        embedding <#> (:qvec)::vector AS distance
                    FROM document_chunks
                    WHERE user_id = :user_id
                    ORDER BY distance ASC
                    LIMIT :k
                """
                params = {
                    "qvec": query_vec,
                    "user_id": user_id,
                    "k": k,
                }
            else:
                sql = """
                    SELECT
                        doc_name,
                        chunk_index,
                        content,
                        embedding <#> (:qvec)::vector AS distance
                    FROM document_chunks
                    WHERE user_id = :user_id
                      AND doc_name = :doc_name
                    ORDER BY distance ASC
                    LIMIT :k
                """
                params = {
                    "qvec": query_vec,
                    "user_id": user_id,
                    "doc_name": doc_name,
                    "k": k,
                }

            rows = self.db.execute(text(sql), params).all()
            logger.info(f"top_k (semantic) returned {len(rows)} rows")
            return rows
        except Exception as exc:
            logger.exception(f"Error in top_k query: {exc}")
            raise

    def search_bm25(self, user_id: int, query: str, k: int, doc_name: str | None):
        """
        BM25-style keyword search using Postgres full-text search.
        Uses 'content_tsv' GIN index and ts_rank_cd for ranking.
        Now also selects chunk_index for better dedup + source tracking.
        """
        logger.info(
            f"BM25 search: user_id={user_id}, doc_name={doc_name}, "
            f"query='{query[:100]}{'...' if len(query) > 100 else ''}'"
        )
        try:
            if doc_name is None:
                sql = """
                    SELECT
                        doc_name,
                        chunk_index,
                        content,
                        ts_rank_cd(content_tsv, plainto_tsquery('english', :q)) AS rank
                    FROM document_chunks
                    WHERE user_id = :user_id
                      AND content_tsv @@ plainto_tsquery('english', :q)
                    ORDER BY rank DESC
                    LIMIT :k
                """
                params = {
                    "q": query,
                    "user_id": user_id,
                    "k": k,
                }
            else:
                sql = """
                    SELECT
                        doc_name,
                        chunk_index,
                        content,
                        ts_rank_cd(content_tsv, plainto_tsquery('english', :q)) AS rank
                    FROM document_chunks
                    WHERE user_id = :user_id
                      AND doc_name = :doc_name
                      AND content_tsv @@ plainto_tsquery('english', :q)
                    ORDER BY rank DESC
                    LIMIT :k
                """
                params = {
                    "q": query,
                    "user_id": user_id,
                    "doc_name": doc_name,
                    "k": k,
                }

            rows = self.db.execute(text(sql), params).all()
            logger.info(f"search_bm25 (keyword) returned {len(rows)} rows")
            return rows
        except Exception as exc:
            logger.exception(f"Error in search_bm25 query: {exc}")
            raise

    def get_chunks_for_doc(self, user_id: int, doc_name: str):
        """
        Fetch all chunks for a given document for a user, ordered by chunk_index.
        Used for 'summarize this document' generic queries.
        """
        logger.info(
            f"get_chunks_for_doc: user_id={user_id}, doc_name={doc_name}"
        )
        try:
            sql = """
                SELECT
                    doc_name,
                    chunk_index,
                    content
                FROM document_chunks
                WHERE user_id = :user_id
                  AND doc_name = :doc_name
                ORDER BY chunk_index ASC
            """
            params = {
                "user_id": user_id,
                "doc_name": doc_name,
            }
            rows = self.db.execute(text(sql), params).all()
            logger.info(
                f"get_chunks_for_doc returned {len(rows)} rows for doc_name={doc_name}"
            )
            return rows
        except Exception as exc:
            logger.exception(f"Error in get_chunks_for_doc query: {exc}")
            raise
