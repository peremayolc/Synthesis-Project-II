import json
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from paths import DATA_DIR

DOCS_PATH = DATA_DIR / "rag_docs" / "rag_documents.jsonl"
INDEX_PATH = DATA_DIR / "rag_docs" / "faiss.index"
META_PATH = DATA_DIR / "rag_docs" / "faiss_meta.pkl"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_docs():
    docs = []
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def main():
    docs = load_docs()
    model = SentenceTransformer(MODEL_NAME)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("wb") as f:
        pickle.dump(docs, f)
    print("Index:", INDEX_PATH)
    print("Metadata:", META_PATH)


if __name__ == "__main__":
    main()
