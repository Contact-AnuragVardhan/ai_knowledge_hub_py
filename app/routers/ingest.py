# ingest.py
from pathlib import Path
import os

from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
)
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..db import get_db, SessionLocal
from .. import schemas
from ..models import IngestJob
from ..services.local_embeddings import (
    extract_text_from_path,
    chunk_and_store,
)
from ..services.vector_store import VectorStore
from app.utils.logging import logger

from app.executor import executor

router = APIRouter(prefix="/api")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def process_ingest_job(job_id: int, filepath: str) -> None:
    """
    Background task: read file, extract text, chunk, embed, store, and
    update job status.
    """
    db = SessionLocal()
    try:
        logger.info(f"[INGEST {job_id}] Worker started, filepath={filepath}")

        job = db.get(IngestJob, job_id)
        if not job:
            logger.error(f"[INGEST {job_id}] Job not found in DB")
            return

        logger.info(f"[INGEST {job_id}] Marking job as processing")
        job.status = "processing"
        db.commit()
        db.refresh(job)

        logger.info(f"[INGEST {job_id}] Extracting text from file...")
        text = extract_text_from_path(filepath, job.doc_name)
        logger.info(f"[INGEST {job_id}] Text extracted, length={len(text)}")

        store = VectorStore(db)

        logger.info(f"[INGEST {job_id}] Calling chunk_and_store")
        chunk_and_store(job.user_id, job.doc_name, text, store)
        logger.info(f"[INGEST {job_id}] chunk_and_store completed")

        job.status = "completed"
        job.error = None
        db.commit()
        logger.info(f"[INGEST {job_id}] Job marked as completed")

    except Exception as exc:
        logger.exception(f"[INGEST {job_id}] Job failed: {exc}")
        try:
            job = db.get(IngestJob, job_id)
            if job:
                job.status = "failed"
                job.error = str(exc)
                db.commit()
        except Exception:
            logger.exception(f"[INGEST {job_id}] Failed to update job status to failed")
    finally:
        db.close()
        try:
            os.remove(filepath)
            logger.info(f"[INGEST {job_id}] Removed temp file {filepath}")
        except OSError:
            logger.warning(f"[INGEST {job_id}] Could not remove temp file {filepath}")



@router.post("/ingest", response_model=schemas.IngestResponse)
async def ingest(
    background_tasks: BackgroundTasks,
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

    # 1) Save uploaded file to disk
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filepath = UPLOAD_DIR / f"{user.id}_{file.filename}"
    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    # 2) Create ingest job
    job = IngestJob(
        user_id=user.id,
        doc_name=file.filename,
        file_path=str(filepath),
        status="pending",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # 3) Schedule background processing
    executor.submit(process_ingest_job, job.id, str(filepath))

    logger.info(
        f"Ingest job {job.id} queued for user_id={user.id}, filename={file.filename}"
    )

    return {
        "name": file.filename,
        "status": "queued",
        "job_id": job.id,
    }


@router.get("/ingest-jobs/{job_id}", response_model=schemas.IngestJobStatus)
def get_ingest_job_status(
    job_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    job = db.get(IngestJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Job not found")

    return schemas.IngestJobStatus(
        id=job.id,
        name=job.doc_name,
        status=job.status,
        error=job.error,
    )
