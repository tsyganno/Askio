from app.services.embeddings import EmbeddingsClient
from app.services.vectorstore_qdrant import QdrantStore
import os


class RAGService:
    def __init__(self):
        self.emb = EmbeddingsClient()
        self.vs = QdrantStore()
        self.llm_provider = os.environ.get("LLM_PROVIDER", "openai")

    async def ingest_document(self, text: str, doc_id: str, metadata: dict):
        # простое разбиение на параграфы/чанки
        chunks = [c for c in text.split("\n\n") if c.strip()][:50]
        embs = await self.emb.embed_text(chunks)
        for i, chunk in enumerate(chunks):
            self.vs.upsert(f"{doc_id}_{i}", embs[i], {"text": chunk, **metadata})

    async def answer(self, question: str):
        q_emb = (await self.emb.embed_text([question]))[0]
        hits = self.vs.search(q_emb, top_k=5)
        context = "\n\n".join([h["text"] for h in hits])
        prompt = f"Use the following context to answer the question. Context:\n{context}\n\nQuestion: {question}\nAnswer concisely and cite sources."
        # call LLM (OpenAI / Anthropic)
        answer = await self._call_llm(prompt)
        # tokens estimation basic
        tokens = len(prompt.split()) + len(answer.split())
        sources = [{"id": h["id"], "score": h["score"]} for h in hits]
        return answer, sources, tokens

    async def _call_llm(self, prompt: str):
        if self.llm_provider == "openai":
            import openai, os
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=512)
            return r["choices"][0]["message"]["content"]
        else:
            # placeholder for Anthropic
            return "LLM response"
