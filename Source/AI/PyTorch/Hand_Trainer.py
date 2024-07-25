import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class handLandmarkData(Dataset):
    def __init__(self,
                 data : np.ndarray,
                 label : np.ndarray):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]

# Load data from npy file, change to csv please.
def readDataset(dataPath : Path,
                labelPath : Path,
                batchSize : int,
                splitRatio : float):
    data, label = np.load(dataPath), np.load(labelPath)
    trainData, testData = train_test_split(data, test_size=splitRatio, random_state=69, shuffle=False)
    trainLabel, testLabel = train_test_split(label, test_size=splitRatio, random_state=69, shuffle=False)

    trainDataset = handLandmarkData(trainData, trainLabel)
    trainDataset = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)

    testDataset = handLandmarkData(testData, testLabel)
    testDataset = DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=True)
    return trainDataset, testDataset

class handLandmarkAnalyzer(nn.Module):
    def __init__(self):
        # 5 x 84 x 3 as input
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3))
        self.maxPooling = nn.MaxPool2d(kernel_size=(2, 2))
        self.flattern = nn.Flatten()
        self.LSTM1 = nn.LSTM(input_size=1260, hidden_size=8, num_layers=2)
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

def trainModel(model : nn.Module,
               epoch : int,
               lossFunction,
               optimizer,
               trainData,
               validationData):
    trainHistory = {
        "trainLoss" : [],
        "trainAccuracy" : [],
        "evaluationLoss" : [],
        "evaluationAccuracy" : []
    }
    for iteration in range(epoch + 1):
        # Training
        trainLoss, correct = 0, 0
        model.train()
        for data, label in trainData:
            data, label = data.to(device), label.to(device)
            prediction = model(data)
            loss = lossFunction(prediction, data)

            optimizer.zero_grad() # Reset loss from last iteration
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()
            correct += (prediction.argmax(1) == label).float().sum().item()
        trainHistory["trainLoss"].append(trainLoss / len(trainData))
        trainHistory["trainAccuracy"].append(correct / len(trainData.dataset))

        # Validation
        validationLoss, correct = 0, 0
        model.eval()
        for data, label in validationData:
            data, label = data.to(device), label.to(device)
            prediction = model(data)
            loss = lossFunction(prediction, data)
            validationLoss += loss.item()
            correct += (prediction.argmax(1) == label).float().sum().item()
        trainHistory["evaluationLoss"].append(validationLoss / len(validationData))
        trainHistory["evaluationAccuracy"].append(correct / len(validationData.dataset))
    return model, trainHistory

model = handLandmarkAnalyzer().to(device)
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batchSize = 32

datasetDir = Path(__file__).parent.joinpath("Data")
trainDataset, testDataset = readDataset(datasetDir.joinpath("Features.npy"),
                                        datasetDir.joinpath("Labels.npy"),
                                        batchSize,
                                        0.2)
model, trainHistory = trainModel(model,
                                 epoch=100,
                                 lossFunction=lossFunction,
                                 optimizer=optimizer,
                                 trainData=trainDataset,
                                 validationData=testDataset)