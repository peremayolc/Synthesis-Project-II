import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from paths import DATA_DIR

INDEX_PATH = DATA_DIR / "rag_docs" / "faiss.index"
META_PATH = DATA_DIR / "rag_docs" / "faiss_meta.pkl"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Retriever:
    def __init__(self):
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "RAG index not found. Run: python src/build_rag_docs.py && python src/rag_index.py"
            )
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_PATH))
        print("Loading metadata...")
        with META_PATH.open("rb") as f:
            self.docs = pickle.load(f)

    def retrieve(self, query, k=3):
        emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(emb.astype(np.float32), k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            doc = self.docs[idx]
            results.append({"score": float(score), "type": doc["type"], "text": doc["text"]})
        return results
