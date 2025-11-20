from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from ..auth import get_current_user
from ..db import get_db
from app.utils.logging import logger

router = APIRouter(prefix="/api")


@router.get("/docs")
def list_docs(db: Session = Depends(get_db), user=Depends(get_current_user)):
    logger.info(f"/docs called for user_id={user.id}, username={user.username}")

    rows = (
        db.execute(
            text(
                "SELECT DISTINCT doc_name FROM document_chunks "
                "WHERE user_id = :uid ORDER BY doc_name"
            ),
            {"uid": user.id},
        )
        .scalars()
        .all()
    )

    logger.info(f"/docs returning {len(rows)} documents for user_id={user.id}")
    return rows
