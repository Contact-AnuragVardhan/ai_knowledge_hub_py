from pydantic_settings import BaseSettings, SettingsConfigDict
from app.utils.logging import logger


class Settings(BaseSettings):
    database_url: str
    openai_api_key: str
    jwt_secret: str
    jwt_algo: str = "HS256"
    chunk_size: int = 3000
    top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore any truly extra env vars
    )


settings = Settings()
logger.info("Settings loaded successfully from .env")
