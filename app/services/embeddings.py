import time
from typing import List
from fastapi import UploadFile
from openai import OpenAI
from pypdf import PdfReader
from docx import Document
from .vector_store import VectorStore
from app.config import settings
from app.utils.logging import logger

client = OpenAI(api_key=settings.openai_api_key)


def extract_text(file: UploadFile) -> str:
    filename = file.filename or "unknown"
    lower = filename.lower()
    logger.info(f"Extracting text from file: {filename}")

    try:
        if lower.endswith(".pdf"):
            logger.debug("Detected PDF file type")
            reader = PdfReader(file.file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif lower.endswith(".docx"):
            logger.debug("Detected DOCX file type")
            doc = Document(file.file)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            logger.debug("Detected text/other file type")
            text = file.file.read().decode("utf-8")

        logger.info(f"Extracted {len(text)} characters from {filename}")
        return text
    except Exception as exc:
        logger.exception(f"Failed to extract text from {filename}: {exc}")
        raise


def embed_text(content: str) -> List[float]:
    logger.info(f"Creating embedding for content length={len(content)}")
    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input=content)
        embedding = resp.data[0].embedding
        logger.debug("Embedding created successfully")
        return embedding
    except Exception as exc:
        logger.exception(f"Embedding creation failed: {exc}")
        raise


def chunk_and_store(user_id: int, doc_name: str, text: str, store: VectorStore):
    logger.info(
        f"Chunking and storing document: user_id={user_id}, doc_name={doc_name}, "
        f"text_length={len(text)}"
    )

    # Safety cap for very large docs
    text = text[:100_000]

    chunks = [
        text[i:i + settings.chunk_size].strip()
        for i in range(0, len(text), settings.chunk_size)
        if text[i:i + settings.chunk_size].strip()
    ]

    logger.info(f"Total chunks to store for '{doc_name}': {len(chunks)}")

    for idx, chunk in enumerate(chunks):
        logger.debug(f"Embedding chunk {idx + 1}/{len(chunks)} for '{doc_name}'")
        vec = embed_text(chunk)
        store.insert_chunk(user_id, doc_name, idx, chunk, vec)
        time.sleep(0.2)

    logger.info(f"Completed chunking and storing for '{doc_name}'")
