# evaluate.py
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from data.load_dataset import load_fiqa_pairs, deduplicate_pairs

MODEL_PATH = "matryoshka-bge-small-finance"
DIMS       = [64, 128, 256, 384]


def evaluate_at_dim(q_emb: np.ndarray,
                    a_emb: np.ndarray,
                    dim: int,
                    k: int = 1) -> tuple[float, float]:
    q_sliced = q_emb[:, :dim]
    a_sliced = a_emb[:, :dim]

    # warmup run
    _ = cosine_similarity(q_sliced, a_sliced)

    # timed runs
    runs  = 100
    start = time.perf_counter()
    for _ in range(runs):
        sim_matrix = cosine_similarity(q_sliced, a_sliced)
    latency_ms = (time.perf_counter() - start) * 1000 / runs

    # recall@k
    count = 0
    for idx, row in enumerate(sim_matrix):
        top_k = np.argsort(row)[::-1][:k]
        if idx in top_k:
            count += 1

    recall = count * 100 / len(sim_matrix)
    return recall, latency_ms


def main():
    model = SentenceTransformer(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

    # load and deduplicate eval set
    all_pairs  = load_fiqa_pairs()
    eval_pairs = all_pairs[100:1600]
    print(f"\nBefore deduplication: {len(eval_pairs)} pairs")
    eval_pairs = deduplicate_pairs(eval_pairs, model, threshold=0.85)
    print(f"After deduplication:  {len(eval_pairs)} pairs\n")

    # encode
    questions    = [p[0] for p in eval_pairs]
    answers      = [p[1] for p in eval_pairs]
    q_embeddings = model.encode(questions, show_progress_bar=True)
    a_embeddings = model.encode(answers,   show_progress_bar=True)

    # evaluate
    print(f"\n{'Dim':<8} {'Recall@1':>10} {'Recall@3':>10} {'Latency(ms)':>14}")
    print("-" * 45)
    for dim in DIMS:
        r1, latency = evaluate_at_dim(q_embeddings, a_embeddings, dim, k=1)
        r3, _       = evaluate_at_dim(q_embeddings, a_embeddings, dim, k=3)
        speedup     = evaluate_at_dim(
            q_embeddings, a_embeddings, DIMS[-1], k=1
        )[1] / latency
        print(f"{dim:<8} {r1:>9.1f}% {r3:>9.1f}% "
              f"{latency:>13.2f}ms  ({speedup:.1f}x faster)")


if __name__ == "__main__":
    main()