from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os


class QdrantStore:
    def __init__(self):
        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=url)
        self.collection = "askio"
        # ensure collection exists
        if "askio" not in [c.name for c in self.client.get_collections().collections]:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    def upsert(self, id: str, vector, payload: dict):
        self.client.upsert(collection_name=self.collection, points=[{
            "id": id,
            "vector": vector,
            "payload": payload
        }])

    def search(self, vector, top_k=5):
        res = self.client.search(collection_name=self.collection, query_vector=vector, limit=top_k)
        # вернуть текстовые фрагменты и метаданные
        return [{"id": p.id, "score": p.score, **p.payload} for p in res]
