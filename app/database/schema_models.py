from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    chunks: int
    tokens: int
    latency_ms: float
