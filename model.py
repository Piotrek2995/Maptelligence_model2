# model.py
import torch
import torch.nn as nn

class WineRegressor(nn.Module):
    def __init__(self, in_features=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # jedna liczba - przewidywana quality
        )

    def forward(self, x):
        return self.net(x)
