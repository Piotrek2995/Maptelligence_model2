# wine_dataset.py
import torch
from torch.utils.data import Dataset

class WineDataset(Dataset):
    def __init__(self, df):
        # df – pełny DataFrame z kolumną "quality"
        features = df.drop("quality", axis=1).values       # X
        labels = df["quality"].values                      # y

        self.X = torch.tensor(features, dtype=torch.float32)
        # traktujemy to jako regresję – jedna liczba (quality)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        # liczba próbek
        return len(self.X)

    def __getitem__(self, idx):
        # zwracamy gotowe tensory
        return self.X[idx], self.y[idx]
