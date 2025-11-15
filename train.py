# train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data_gen import get_dataloaders
from model import WineRegressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Używam urządzenia:", DEVICE)

def train(num_epochs=50, lr=1e-3, batch_size=64):
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = WineRegressor(in_features=11).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoka {epoch}/{num_epochs}"):
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # --- VAL ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoka {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # --- TEST ---
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            test_loss += loss.item() * X_batch.size(0)

    test_loss /= len(test_loader.dataset)
    print(f"\nTest MSE: {test_loss:.4f}")

    return model

if __name__ == "__main__":
    trained_model = train()
