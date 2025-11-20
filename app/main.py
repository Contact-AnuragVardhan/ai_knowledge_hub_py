from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .routers import auth, ingest, docs, query
from .db import Base, engine
from app.utils.logging import logger

# Create all tables
logger.info("Creating database tables (if not exist)")
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Knowledge Hub (FastAPI)")
logger.info("FastAPI app instance created")


# Routers
app.include_router(auth.router)
app.include_router(ingest.router)
app.include_router(docs.router)
app.include_router(query.router)
logger.info("Routers registered: auth, ingest, docs, query")


@app.get("/app")
def hello():
    logger.info("Health check /app endpoint called")
    return "Hello from FastAPI!"


# Global exception handler (nice for logging unexpected errors)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on path {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
