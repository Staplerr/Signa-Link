import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def readDataset(datasetPath: Path):
    return

class handLandmarkAnalyzer(nn.Module):
    def __init__(self):
        # 5 x 84 x 3 as input
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3))
        self.maxPooling = nn.MaxPool2d(kernel_size=(2, 2))
        self.flattern = nn.Flatten()
        self.LSTM1 = nn.LSTM(input_size=1260, num_layers=2)
        self.fcc1 = nn.Linear(in_features=1260, out_features=256)
        self.relu1 = nn.ReLU()
        self.fcc2 = nn.Linear(in_features=256, out_features=8)
        self.softmax1 = nn.Softmax()

    def forward(self, x):
        x = self.conv(x)
        x = self.maxPooling(x)
        x = self.flattern(x)
        x = self.LSTM1(x)
        x = self.fcc1(x)
        x = self.relu1(x)
        x = self.fcc2(x)
        x = self.softmax1(x)
        return x

def trainModel(model, epoch, lossFunction, optimizer, trainData, testData):
    pass

model = handLandmarkAnalyzer().to(device)
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batchSize = 32

model, trainHistory = trainModel(model)