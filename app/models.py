from sqlalchemy import Column, Integer, String, Text, Index
from pgvector.sqlalchemy import VECTOR
from .db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    doc_name = Column(String(255), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(VECTOR(dim=1536))  # match text-embedding-3-small dims
    __table_args__ = (
        Index("ix_chunks_user_doc_idx", "user_id", "doc_name", "chunk_index"),
    )
