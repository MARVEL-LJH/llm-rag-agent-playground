from typing import List
from langchain_core.embeddings import Embeddings
import numpy as np

class SimpleEmbedding(Embeddings):
    """简单模拟Embedding，仅用于调试跑通RAG流程，无真实语义能力"""
    def __init__(self):
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = []
        for t in texts:
            vec = np.random.rand(128).tolist()
            res.append(vec)
        return res

    def embed_query(self, text: str) -> List[float]:
        return np.random.rand(128).tolist()

    def __call__(self, text: str):
        return self.embed_query(text)
