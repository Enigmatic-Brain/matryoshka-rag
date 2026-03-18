from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_fiqa_pairs() -> list[tuple]:
    corpus = []

    print(f"Loading fiqa dataset....")
    fiqa_data = load_dataset("LLukas22/fiqa")
    for row in fiqa_data["train"]:
        question, answer = row.get("question", ""), row.get("answer", "")
        if len(question.strip()) > 10 and len(answer.strip()) > 10:
            corpus.append((question, answer))

    print(f"corpus length after loading fiqa data: {len(corpus)}")

    print(f"Loading FinGPT data....")
    fingpt_data = load_dataset("FinGPT/fingpt-fiqa_qa")
    for row in fingpt_data["train"]:
        question, answer = row.get("input", ""), row.get("output", "")
        if len(question.strip()) > 10 and len(answer.strip()) > 10:
            corpus.append((question, answer))

    print(f"corpus length after loading FinGPT data: {len(corpus)}")
    print("\nSample pairs:")
    for q, a in corpus[:3]:
        print(f"Q: {q[:100]}")
        print(f"A: {a[:100]}")
        print("---")
    
    return corpus


pairs = load_fiqa_pairs()
print(f"\n Total pairs: {len(pairs)}")


def deduplicate_pairs(pairs: list[tuple],
                      model: SentenceTransformer,
                      threshold: float = 0.85) -> list[tuple]:
    questions = [p[0] for p in pairs]
    q_emb = model.encode(questions, show_progress_bar=False)
    sim_matrix = cosine_similarity(q_emb)

    keep = []
    dropped = set()
    for i in range(len(pairs)):
        if i in dropped:
            continue
        keep.append(i)
        for j in range(i + 1, len(pairs)):
            if sim_matrix[i][j] > threshold:
                dropped.add(j)

    print(f"Deduplication: {len(pairs)} → {len(keep)} pairs "
          f"({len(pairs)-len(keep)} removed at threshold {threshold})")
    return [pairs[i] for i in keep]