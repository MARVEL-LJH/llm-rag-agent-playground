from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class SimpleEmbedding(Embeddings):
    def __init__(self):
        # 轻量开源向量模型，第一次运行自动下载
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeds = self.model.encode(texts).tolist()
        return embeds

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def __call__(self, text: str):
        return self.embed_query(text)
