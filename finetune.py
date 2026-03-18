## finetuning.py
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from data.load_dataset import load_fiqa_pairs
from torch.optim import AdamW


model = SentenceTransformer("BAAI/bge-small-en-v1.5")
pairs = load_fiqa_pairs()

## Convert pairs to a list of InputExample objects
train_examples = [InputExample(texts=[pair[0], pair[1]]) for pair in pairs]

## Initialising the loss function
base_loss = MultipleNegativesRankingLoss(model)
loss_function = MatryoshkaLoss(model, base_loss, matryoshka_dims=[64, 128, 256, 384])

train_dataloader = DataLoader(train_examples, batch_size=32, shuffle=True, collate_fn=model.smart_batching_collate)
print(f"Total training examples: {len(train_examples)}")
print(f"Sample: {train_examples[0].texts}")

EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0 
    for batch in train_dataloader:
        optimizer.zero_grad()
        features, labels = batch
        
        loss = loss_function(features, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_dataloader):.4f}")