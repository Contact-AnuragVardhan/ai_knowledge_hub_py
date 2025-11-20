from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from ..auth import get_current_user
from ..db import get_db
from ..services.embeddings import extract_text, chunk_and_store
from ..services.vector_store import VectorStore
from .. import schemas
from app.utils.logging import logger

router = APIRouter(prefix="/api")


@router.post("/ingest", response_model=schemas.IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    if not file:
        logger.warning("Ingest failed: file missing")
        raise HTTPException(status_code=400, detail="file missing")

    logger.info(
        f"Ingest called by user_id={user.id}, username={user.username}, "
        f"filename={file.filename}"
    )

    text = extract_text(file)
    store = VectorStore(db)
    chunk_and_store(user.id, file.filename, text, store)

    logger.info(
        f"Ingest completed for user_id={user.id}, filename={file.filename} "
        f"(indexed into pgvector)"
    )

    return {"name": file.filename, "status": "Indexed into pgvector"}
