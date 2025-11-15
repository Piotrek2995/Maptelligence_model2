import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class WineDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        x = torch.tensor(features.values, dtype=torch.float32)
        y = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
        # upewnij się, że typy są poprawne

    def __len__(self):
        return len(self.features)
        # zwróć liczbę elementów

    def __getitem__(self, idx):
        return self.features.iloc[idx], self.labels.iloc[idx]
        # zwróć (X[idx], y[idx])


dataset = WineDataset(x, y)