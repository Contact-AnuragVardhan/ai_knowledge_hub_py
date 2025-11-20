from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import settings
from app.utils.logging import logger


engine = create_engine(settings.database_url, echo=False, future=True)
logger.info("Database engine created")

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    logger.debug("DB session created")
    try:
        yield db
    except Exception as exc:
        logger.exception(f"Error during DB session usage: {exc}")
        raise
    finally:
        db.close()
        logger.debug("DB session closed")
