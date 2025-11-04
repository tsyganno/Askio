from pydantic import BaseModel
from typing import List


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    tokens: int
    latency_ms: float
    sources: List[str]
