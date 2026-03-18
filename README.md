# MatryoshkaRAG 🪆

A financial domain embedding model fine-tuned with Matryoshka Representation Learning (MRL), enabling dynamic dimension truncation at inference time. The same model serves both a high-throughput low-latency path and a high-accuracy path — no retraining required.

Built on `BAAI/bge-small-en-v1.5` (33M parameters), fine-tuned using contrastive learning on financial QA pairs.

---

## The core idea

Standard embedding models produce a fixed-size vector. You always get 384 dimensions whether you need them or not. Matryoshka Representation Learning trains the model so that the **first `d` dimensions are already a good embedding for any `d`** — you can truncate at inference time and trade accuracy for speed dynamically.

```
Standard:    always 384 dims → one fixed accuracy/latency point
Matryoshka:  truncate to 64, 128, 256, or 384 → choose your tradeoff at runtime
```

This is the technique behind OpenAI's `text-embedding-3` models (Kusupati et al., NeurIPS 2022).

---

## Results

Evaluated on 611 deduplicated held-out financial QA pairs from FIQA:

| Dimensions | Recall@1 | Recall@3 | Latency  | Speedup vs 384 |
|------------|----------|----------|----------|----------------|
| 64         | 33.9%    | 48.0%    | 1.28ms   | 2.4x faster    |
| 128        | 44.7%    | 62.4%    | 1.72ms   | 1.8x faster    |
| 256        | 53.8%    | 70.0%    | 2.53ms   | 1.2x faster    |
| **384**    | **57.1%**| **73.5%**| **3.06ms**| baseline      |

*Recall@1: correct answer is top result. Recall@3: correct answer is in top 3 results.*
*Latency averaged over 100 runs on GPU.*

**Key finding:** 128 dimensions achieves 78% of full-model Recall@1 at 1.8x lower
latency — same model, same weights, truncate at inference time.

### Dataset quality finding

59% of FIQA's questions were near-duplicates at cosine similarity threshold 0.85
(1500 raw pairs → 611 after deduplication). Strict Recall@1 without deduplication
underestimates model quality by approximately 23 percentage points. This is a
systematic evaluation bias in standard IR benchmarks that is rarely acknowledged.

---

## Why this matters for production RAG

In a RAG system, every query requires:
1. Embedding the query
2. Computing similarity against thousands of stored document vectors
3. Returning top-k matches

Step 2 scales with embedding dimension. At 1000 queries/second against 100k documents:

```
384 dims: 3.06ms × 1000 = 3.06 seconds of similarity computation per second
128 dims: 1.72ms × 1000 = 1.72 seconds — 44% reduction in compute
 64 dims: 1.28ms × 1000 = 1.28 seconds — 58% reduction in compute
```

A single trained model can serve multiple latency SLAs without redeployment.

---

## Training

### Model
`BAAI/bge-small-en-v1.5` — 33M parameters, 384 output dimensions, English.
Chosen for CPU/single-GPU trainability while maintaining competitive performance
on financial text.

### Data
Two open-source financial QA datasets:

| Dataset | Rows | Content |
|---------|------|---------|
| [LLukas22/fiqa](https://huggingface.co/datasets/LLukas22/fiqa) | ~6k | Financial opinion QA |
| [FinGPT/fingpt-fiqa_qa](https://huggingface.co/datasets/FinGPT/fingpt-fiqa_qa) | ~17k | Financial QA pairs |

Each `(question, answer)` pair is a **positive pair** for contrastive training.
Negative pairs are constructed implicitly — every other answer in the batch acts
as a negative for each question. No manual negative curation required.

### Loss function
`MatryoshkaLoss` wrapping `MultipleNegativesRankingLoss` from `sentence-transformers`:

```python
base_loss     = MultipleNegativesRankingLoss(model)
loss_function = MatryoshkaLoss(model, base_loss, matryoshka_dims=[64, 128, 256, 384])
```

At each training step, MNRL loss is computed at all four dimension slices
simultaneously and summed. Backpropagation through all four losses forces the
model to pack the most important semantic information into the earliest dimensions.

### Hyperparameters
```
Base model   : BAAI/bge-small-en-v1.5
Epochs       : 3
Batch size   : 32
Learning rate: 2e-5(AdamW)
Warmup steps : 10% of total steps
Matryoshka dims: [64, 128, 256, 384]
Final loss   : 1.02 (MNRL at 384 dims)
```

### Why MultipleNegativesRankingLoss

MNRL treats every other answer in the batch as an implicit negative for each
question. With batch size 32, each question sees 31 negatives per step — no
manual negative mining required. The loss is cross-entropy over a (B × B)
similarity matrix where the diagonal should be highest.

Larger batches = harder negatives = stronger training signal. This is why
batch size is the most important hyperparameter for contrastive embedding training.

---

## Architecture — what MRL changes

Standard fine-tuning trains the model to produce good 384-dim embeddings.
MRL changes only the **loss function** — the model architecture is unchanged.

```
Input sentence
      ↓
  bge-small transformer (33M params, unchanged architecture)
      ↓
  Mean pooling → (batch_size, 384)
      ↓
  MatryoshkaLoss slices at [64, 128, 256, 384]
      ↓
  MNRL loss computed at each slice
      ↓
  Total loss = L_64 + L_128 + L_256 + L_384
      ↓
  Backprop through all four — forces nested representations
```

At inference, simply truncate the output vector:

```python
embedding = model.encode("What is EBITDA?")  # (384,)
embedding_128 = embedding[:128]               # (128,) — valid embedding
embedding_64  = embedding[:64]               # (64,)  — valid embedding
```

---

## Usage

### Install

```bash
pip install sentence-transformers datasets
```

### Encode at any dimension

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("matryoshka-bge-small-finance")

sentences = [
    "What is EBITDA?",
    "How does the Federal Reserve set interest rates?",
    "What caused the 2008 financial crisis?",
]

# full 384-dim embeddings
embeddings_384 = model.encode(sentences)

# truncate to 128 dims for faster retrieval
embeddings_128 = embeddings_384[:, :128]

print(f"Full embeddings: {embeddings_384.shape}")   # (3, 384)
print(f"Fast embeddings: {embeddings_128.shape}")   # (3, 128)
```

### RAG retrieval at variable dimension

```python
from sklearn.metrics.pairwise import cosine_similarity

def retrieve(query, corpus_embeddings, corpus_texts, dim=384, top_k=3):
    query_emb = model.encode([query])[:, :dim]
    corpus_emb = corpus_embeddings[:, :dim]
    scores = cosine_similarity(query_emb, corpus_emb)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(corpus_texts[i], scores[i]) for i in top_indices]

# same function, different speed/accuracy tradeoff
results_fast     = retrieve(query, corpus_emb, texts, dim=128)
results_accurate = retrieve(query, corpus_emb, texts, dim=384)
```

---

## File structure

```
matryoshka-rag/
├── finetune.py          # MRL fine-tuning pipeline
├── evaluate.py          # Recall@1, Recall@3, latency at all dims
├── rag.py               # FAISS retrieval at variable dimension
├── matryoshka-bge-small-finance/  # saved model weights
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
└── README.md
```

---

## Simplifications vs production

| Aspect | This project | Production |
|--------|-------------|------------|
| Similarity search | sklearn cosine_similarity | FAISS index (10-100x faster) |
| Negatives | In-batch random | Hard negative mining |
| Eval metric | Recall@1, Recall@3 | NDCG@10, MRR |
| Deduplication | Cosine threshold | MinHash LSH for scale |
| Model size | 33M params | 335M+ for higher accuracy |

The most impactful missing piece is **hard negative mining** — finding negatives
that are semantically close to the query but not the correct answer. Random
in-batch negatives are easy to distinguish; hard negatives force the model to
learn finer-grained distinctions. This is how production embedding models like
`text-embedding-3` achieve much higher Recall@1.

---

## Evaluation methodology

### Why we deduplicated

Standard Recall@1 assumes each question has exactly one correct answer. FIQA
contains near-duplicate questions — semantically identical queries phrased
differently. When the model retrieves the correct answer for a duplicate, it
scores as a false negative. Deduplication at cosine threshold 0.85 removes
this systematic bias.

```python
# 1500 raw pairs → 611 after deduplication
# 59% of questions were near-duplicates
eval_pairs = deduplicate_pairs(raw_pairs, model, threshold=0.85)
```

### Why we use Recall@3 alongside Recall@1

In production RAG, the retrieved chunks are passed to an LLM which can reason
across multiple passages. Recall@3 — whether the correct answer appears in the
top 3 results — is often more representative of end-to-end RAG quality than
strict Recall@1.

---

## References

- Kusupati, A. et al. (2022). [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147). NeurIPS 2022. — *Original MRL paper*
- Xiao, S. et al. (2023). [C-Pack: Packaged Resources To Advance General Chinese Embedding](https://arxiv.org/abs/2309.07597). — *BGE model family*
- Chen, T. et al. (2020). [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709). ICML 2020. — *SimCLR: foundational contrastive learning*
- Thakur, N. et al. (2021). [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation](https://arxiv.org/abs/2104.08663). NeurIPS 2021. — *FIQA benchmark source*
- Reimers, N. & Gurevych, I. (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084). EMNLP 2019. — *sentence-transformers library*

---

## What I learned

The most surprising finding was the dataset quality issue. FIQA is a widely cited
IR benchmark used in dozens of papers. Running cosine similarity across the
question set revealed that 59% of questions were near-duplicates — semantically
identical queries phrased differently. This means every paper reporting Recall@1
on raw FIQA is measuring a combination of model quality and annotation noise.

The technical lesson: **always inspect your evaluation set before trusting your
metrics.** A model that scores 36% on raw FIQA and 57% on deduplicated FIQA
hasn't changed — the measurement has.

The MRL lesson: the jump from 64 to 128 dimensions (+10.8 Recall@1 points) is
far larger than the jump from 128 to 384 (+12.4 points spread over 256 extra
dimensions). The first 128 dimensions carry disproportionately more semantic
information than the remaining 256. This is the core Matryoshka property — and
it means that for most production use cases, 128 dimensions is the sweet spot.
