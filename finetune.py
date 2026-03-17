## finetuning.py
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from data.load_dataset import load_fiqa_pairs


model = SentenceTransformer("BAAI/bge-small-en-v1.5")
pairs = load_fiqa_pairs()

## Convert pairs to a list of InputExample objects
train_examples = [InputExample(texts=[pair[0], pair[1]]) for pair in pairs]

## Initialising the loss function
base_loss = MultipleNegativesRankingLoss(model)
loss_function = MatryoshkaLoss(model, base_loss, matryoshka_dims=[64, 128, 256, 384])

train_dataloader = DataLoader(train_examples, batch_size=32, shuffle=True)
print(f"Total training examples: {len(train_examples)}")
print(f"Sample: {train_examples[0].texts}")

epochs = 3
warmup_steps = int(0.1 * (len(train_dataloader) * epochs))
model.fit(
    train_objectives=[(train_dataloader, loss_function)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path="matryoshka-bge-small-finance",
    show_progress_bar=True,
)
