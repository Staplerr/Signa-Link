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
        super().__init__()

    def forward(self, x):
        return x
    
def trainModel(model, epoch, lossFunction, optimizer, trainData, testData):
    pass

model = handLandmarkAnalyzer().to(device)
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batchSize = 32

model, trainHistory = trainModel(model)