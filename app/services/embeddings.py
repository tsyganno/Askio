import os
import openai
from typing import List


class EmbeddingsClient:
    def __init__(self):
        self.provider = os.environ.get("LLM_PROVIDER","openai")
        self.openai_key = os.environ.get("OPENAI_API_KEY")

    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        # Пример: использовать openai embeddings (synchronous -> обернуть в threadpool если async)
        openai.api_key = self.openai_key
        out = []
        for t in texts:
            r = openai.Embedding.create(model="text-embedding-3-small", input=t)
            out.append(r["data"][0]["embedding"])
        return out
