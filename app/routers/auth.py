from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import schemas
from ..db import get_db
from ..models import User
from ..auth import hash_password, verify_password, create_token
from app.utils.logging import logger

router = APIRouter(prefix="/api")


@router.post("/register", response_model=schemas.RegisterResponse)
def register(payload: schemas.RegisterRequest, db: Session = Depends(get_db)):
    logger.info(f"Register endpoint called for username={payload.username}")

    existing = db.query(User).filter_by(username=payload.username).first()
    if existing:
        logger.warning(f"Register failed: user already exists username={payload.username}")
        raise HTTPException(status_code=400, detail="user exists")

    user = User(username=payload.username, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()

    logger.info(f"User registered successfully username={payload.username}")
    return {"status": "registered"}


@router.post("/login", response_model=schemas.LoginResponse)
def login(payload: schemas.LoginRequest, db: Session = Depends(get_db)):
    logger.info(f"Login endpoint called for username={payload.username}")

    user = db.query(User).filter_by(username=payload.username).first()
    if not user:
        logger.warning(f"Login failed: user not found username={payload.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
        )

    if not verify_password(payload.password, user.password_hash):
        logger.warning(f"Login failed: wrong password username={payload.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
        )

    token = create_token(user.id)
    logger.info(f"Login successful username={payload.username}")
    return {"status": "ok", "token": token}
