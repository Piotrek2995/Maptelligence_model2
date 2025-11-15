# data_gen.py
import torch
from torch.utils.data import random_split, DataLoader
from data_verifier import df
from wine_dataset import WineDataset

def get_datasets(seed=42):
    generator = torch.Generator().manual_seed(seed)

    full_ds = WineDataset(df)   # <-- tu pakujemy cały DataFrame w Dataset

    n = len(full_ds)
    train_len = int(0.8 * n)
    val_len = int(0.1 * n)
    test_len = n - train_len - val_len

    train_ds, val_ds, test_ds = random_split(
        full_ds, [train_len, val_len, test_len], generator=generator
    )
    return train_ds, val_ds, test_ds


def get_dataloaders(batch_size=64, seed=42):
    train_ds, val_ds, test_ds = get_datasets(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Liczba batchy w train: {len(train_loader)}")
    print(f"Liczba batchy w val:   {len(val_loader)}")
    print(f"Liczba batchy w test:  {len(test_loader)}")
