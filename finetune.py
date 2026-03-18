import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import (
    MultipleNegativesRankingLoss,
    MatryoshkaLoss,
)
from tqdm import tqdm
from data.load_dataset import load_fiqa_pairs

# --- config ---
MODEL_NAME   = "BAAI/bge-small-en-v1.5"
OUTPUT_PATH  = "matryoshka-bge-small-finance"
EPOCHS       = 3
BATCH_SIZE   = 32
LR           = 2e-5
MATRYOSHKA_DIMS = [64, 128, 256, 384]


def main():
    # load data
    pairs = load_fiqa_pairs()
    print(f"Total training pairs: {len(pairs)}")

    # prepare training examples
    train_examples = [InputExample(texts=[q, a]) for q, a in pairs]

    # load model
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model device: {model.device}")

    # dataloader
    train_dataloader = DataLoader(
        train_examples,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=model.smart_batching_collate,
    )

    # loss
    base_loss     = MultipleNegativesRankingLoss(model)
    loss_function = MatryoshkaLoss(
        model, base_loss, matryoshka_dims=MATRYOSHKA_DIMS
    )

    # optimizer and warmup
    optimizer    = AdamW(model.parameters(), lr=LR)
    warmup_steps = int(0.1 * len(train_dataloader) * EPOCHS)
    device       = model.device

    # training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            unit="batch"
        )

        for batch in progress_bar:
            optimizer.zero_grad()
            features, labels = batch
            labels   = labels.to(device)
            features = [
                {k: v.to(device) for k, v in f.items()}
                for f in features
            ]

            loss = loss_function(features, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg':  f"{total_loss / (progress_bar.n + 1):.4f}"
            })

        epoch_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1}/{EPOCHS} — Avg Loss: {epoch_loss:.4f}\n")

        # save checkpoint after each epoch
        model.save(f"{OUTPUT_PATH}-epoch-{epoch+1}")

    # save final model
    model.save(OUTPUT_PATH)
    print(f"Model saved to {OUTPUT_PATH}")