from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from passlib.hash import bcrypt
from sqlalchemy.orm import Session
from .db import get_db
from .models import User
from app.config import settings
from app.utils.logging import logger

security = HTTPBearer()

SECRET = settings.jwt_secret
ALGO = settings.jwt_algo


def hash_password(pwd: str) -> str:
    logger.debug("Hashing password")
    #TODO: revert this later
    #return bcrypt.hash(pwd)
    return pwd


def verify_password(pwd: str, hashed: str) -> bool:
    logger.debug("Verifying password hash")
    #TODO: revert this later
    #return bcrypt.verify(pwd, hashed)
    return pwd == hashed


def create_token(user_id: int) -> str:
    logger.info(f"Creating JWT token for user_id={user_id}")
    return jwt.encode({"sub": str(user_id)}, SECRET, algorithm=ALGO)


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    token = creds.credentials
    logger.debug("Decoding JWT token for current user")

    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError) as exc:
        logger.warning(f"Invalid token: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    user = db.get(User, user_id)
    if not user:
        logger.warning(f"Token refers to non-existent user_id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    logger.debug(f"Authenticated user_id={user_id}, username={user.username}")
    return user
