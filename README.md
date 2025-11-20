
# AI Knowledge Hub – Python Backend (FastAPI + pgvector + BM25 Hybrid RAG)

This backend powers an AI Knowledge Hub with features such as user authentication, document ingestion, semantic search using pgvector, BM25 full-text search, and hybrid RAG for answering user queries.

## Features
- User registration and JWT-based login
- Document ingestion (PDF, DOCX, TXT)
- Text extraction & chunking
- OpenAI embeddings
- Semantic search (pgvector)
- BM25 keyword search
- Hybrid RAG (BM25 + Semantic)
- GPT-4o-mini answer generation
- Doc-specific filtering via `doc_name`

## Tech Stack
- FastAPI
- PostgreSQL + pgvector
- OpenAI API
- SQLAlchemy ORM
- Full-text search (Postgres FTS)
- Python 3.10+

## Database Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
   id SERIAL PRIMARY KEY,
   username TEXT UNIQUE NOT NULL,
   password_hash TEXT NOT NULL
);

CREATE TABLE document_chunks (
     id SERIAL PRIMARY KEY,
     user_id INT REFERENCES users(id),
     doc_name TEXT,
     chunk_index INT,
     content TEXT,
     embedding vector(1536),
     content_tsv tsvector GENERATED ALWAYS AS (
       to_tsvector('english', coalesce(content, ''))
     ) STORED
);

CREATE INDEX idx_document_chunks_tsv
  ON document_chunks USING GIN (content_tsv);
```

## Installation

```
pip install -r requirements.txt
```

## Environment Variables

```
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/aihub
OPENAI_API_KEY=your_openai_key
SECRET_KEY=your_jwt_secret
CHUNK_SIZE=2000
TOP_K=5
```

## Running the Server

```
uvicorn app.main:app --reload
```

Open API docs:

```
http://localhost:8000/docs
```

## API Endpoints

### Register
```
POST /api/register
{
  "username": "test",
  "password": "123"
}
```

### Login
```
POST /api/login
→ returns token
```

### Ingest Document
```
POST /api/ingest
file: <upload>
```

### Query
#### Search all documents:
```
POST /api/query
{
  "query": "What is the policy?"
}
```

#### Search inside a specific document:
```
POST /api/query
{
  "query": "Explain billing",
  "doc_name": "billing.pdf"
}
```

## Hybrid Search Flow

```
User Query
  → Embedding (semantic)
  → pgvector semantic search
  → BM25 keyword search
  → Merge & dedupe
  → RAG prompt
  → GPT-4o-mini response
```

## Folder Structure

```
app/
 ├── api/
 │    ├── auth.py
 │    ├── docs.py
 │    ├── ingest.py
 │    └── query.py
 ├── services/
 │    ├── embeddings.py
 │    ├── rag.py
 │    └── vector_store.py
 ├── models/
 ├── db.py
 ├── schemas.py
 ├── config.py
 └── utils/
      └── logging.py
```

## License
MIT License.
