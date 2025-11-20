# app/utils/logging.py

import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# Create formatter
formatter = logging.Formatter(LOG_FORMAT)

# ---------------------------
# Console Handler (PyCharm safe)
# ---------------------------
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# ---------------------------
# Rotating File Handler
# ---------------------------
file_handler = RotatingFileHandler(
    f"{LOG_DIR}/app.log",
    maxBytes=5 * 1024 * 1024,   # 5MB
    backupCount=5,
    encoding="utf-8",
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# ---------------------------
# Root Logger Configuration
# ---------------------------
logger = logging.getLogger("aihub")
logger.setLevel(logging.INFO)

# Important: avoid adding duplicate handlers when reloading (Uvicorn reload)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Prevent double logging through uvicorn / fastapi loggers
logger.propagate = False
