from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..db import get_db
from ..services.vector_store import VectorStore
from ..services.rag import answer_query
from .. import schemas

from app.utils.logging import logger

router = APIRouter(prefix="/api")


@router.post("/query", response_model=schemas.QueryResponse)
def query(
    payload: schemas.QueryRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    logger.info(
        f"/query called by user_id={user.id}, username={user.username}, "
        f"docName={payload.docName}, query='{payload.query[:100]}{'...' if payload.query and len(payload.query) > 100 else ''}'"
    )

    if not payload.query:
        logger.warning("Query rejected: 'query' field missing in payload")
        raise HTTPException(status_code=400, detail="query missing")

    try:
        store = VectorStore(db)

        answer, sources = answer_query(
            store, user.id, payload.docName, payload.query
        )

        logger.info(
            f"/query response ready for user_id={user.id}: "
            f"answer_len={len(answer)}, sources={len(sources)}"
        )

        return {"answer": answer, "sources": sources}

    except Exception as exc:
        logger.exception(f"Error processing /query request: {exc}")
        raise HTTPException(status_code=500, detail="Failed to process query")
