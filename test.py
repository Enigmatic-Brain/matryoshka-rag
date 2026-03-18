from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("output")
test_sentences = [
    "What is EBITDA?",
    "EBITDA stands for Earnings Before Interest Taxes Depreciation and Amortisation",
    "The Federal Reserve raised interest rates",
    "What is the yield curve?"
]

embeddings = model.encode(test_sentences)

sim_matrix = cosine_similarity(embeddings)
print(np.round(sim_matrix, 2))

## You should see output like this:
# [[1.   0.82 0.03 0.11]
#  [0.82 1.   0.15 0.13]
#  [0.03 0.15 1.   0.44]
#  [0.11 0.13 0.44 1.  ]]
