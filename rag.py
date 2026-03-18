# rag.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from data.load_dataset import load_fiqa_pairs

MODEL_PATH = "matryoshka-bge-small-finance"


class MatryoshkaRAG:
    """
    RAG retrieval system using a Matryoshka embedding model.
    Supports dynamic dimension truncation at query time.
    """

    def __init__(self, model_path: str, dim: int = 384):
        self.model = SentenceTransformer(model_path)
        self.dim   = dim
        self.corpus_embeddings = None
        self.corpus_texts      = None
        print(f"Loaded model from {model_path} — using {dim} dims")

    def index(self, texts: list[str]):
        """Encode and store corpus embeddings."""
        print(f"Indexing {len(texts)} documents at {self.dim} dims...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.corpus_embeddings = embeddings[:, :self.dim]
        self.corpus_texts      = texts
        print("Indexing complete.")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top-k most relevant documents for a query."""
        if self.corpus_embeddings is None:
            raise RuntimeError("Call index() before retrieve()")

        query_emb = self.model.encode([query])[:, :self.dim]
        scores    = cosine_similarity(query_emb, self.corpus_embeddings)[0]
        top_idx   = np.argsort(scores)[::-1][:top_k]

        return [
            {"text": self.corpus_texts[i], "score": float(scores[i])}
            for i in top_idx
        ]


def main():
    # build a small demo knowledge base from FIQA answers
    pairs   = load_fiqa_pairs()[:500]
    answers = [p[1] for p in pairs]

    # demo at two different dimensions
    for dim in [128, 384]:
        print(f"\n{'='*60}")
        print(f"RAG demo at {dim} dimensions")
        print('='*60)

        rag = MatryoshkaRAG(MODEL_PATH, dim=dim)
        rag.index(answers)

        queries = [
            "What is EBITDA and why does it matter?",
            "How does the Federal Reserve control inflation?",
            "What caused the 2008 financial crisis?",
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            results = rag.retrieve(query, top_k=2)
            for i, r in enumerate(results):
                print(f"  [{i+1}] score={r['score']:.3f} "
                      f"| {r['text'][:100]}...")


if __name__ == "__main__":
    main()