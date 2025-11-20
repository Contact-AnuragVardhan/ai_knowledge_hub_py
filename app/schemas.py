from pydantic import BaseModel
from typing import List, Optional

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    status: str = "ok"
    token: str

class RegisterResponse(BaseModel):
    status: str = "registered"

class IngestResponse(BaseModel):
    name: str
    status: str

class QueryRequest(BaseModel):
    query: str
    docName: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
