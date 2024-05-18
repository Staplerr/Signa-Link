import tensorflow as tf
import keras
from keras import layers
from tensorflow import nn
import pandas as pd
from pathlib import Path
import numpy as np
import time
import configparser

parentDirectory = Path(__file__).parent
labelList = {"กรอบ": 0,     "กระเพรา": 1,    "ขา": 2,       "ข้าว": 3,
             "ไข่": 4,       "คะน้า": 5,      "เค็ม": 6,       "โจ๊ก": 7,
             "แดง": 8,      "ต้ม": 9,        "แตงโม": 10,    "น้ำพริกเผา": 11,
             "บะหมี่": 12,    "เปรี้ยว": 13,    "ผัด": 14,       "ฝรั่ง": 15,
             "พริกแกง": 16,  "มะม่วง": 17,    "ม้า": 18,       "มาม่า": 19,
             "ลูกชิ้นปลา": 20, "เลือด": 21,     "สับ": 22,       "เส้นเล็ก": 23,
             "เส้นใหญ่": 24,  "หมู": 25,       "หวาน": 26,     "องุ่น": 27,
             "แอปเปิ้ล": 28}
configFilePath = parentDirectory.joinpath("config.cfg")
if not configFilePath.exists():
    raise Exception("No config file found")
configFile = configparser.RawConfigParser()
configFile.read(configFilePath)

#Change preference in config.cfg
policy = configFile['Options']['policy']
batchSize = int(configFile['Options']['batchSize'])

keras.mixed_precision.set_global_policy(policy)
outputFolderName = parentDirectory.joinpath(f"Matrix model/matrix_model_{policy}")
outputFolderName.mkdir(parents=True, exist_ok=True)

def preprocessData(dataset):
    label = dataset["Label"].values
    data = dataset.drop(["Label"], axis=1)
    data = data.to_numpy().reshape((-1, len(data.columns))) #sperate ndarray into multiple one corresponding to its label
    return data, label

def splitData(data, label, ratio):
    trainData = data[0:int(ratio * len(data))]
    testData = data[int(ratio * len(data)):-1]

    trainLabel = label[0:int(ratio * len(label))]
    testLabel = label[int(ratio * len(label)):-1]
    return trainData, testData, trainLabel, testLabel

#Preparing dataset
dataset = pd.read_csv(parentDirectory.joinpath("Dataset.csv"))
loadStart = time.perf_counter()
data, label = preprocessData(dataset)
trainData, testData, trainLabel, testLabel = splitData(data, label, 0.8)
loadFinish = time.perf_counter()
print(f"Dataset load time: {loadFinish - loadStart}")
print(f"Total data in train dataset: {len(trainData)}, Total data in test dataset: {len(testData)} ")

model = keras.models.Sequential([
    layers.InputLayer(2010),
    layers.Dense(1024, activation=nn.relu),
    layers.Dropout(0.5),
    layers.Dense(512, activation=nn.relu),
    layers.Dropout(0.5),
    layers.Dense(256, activation=nn.relu),
    layers.Dropout(0.5),
    layers.Dense(128, activation=nn.relu),
    layers.Dropout(0.5),
    layers.Dense(len(labelList), activation=nn.softmax)
])
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()
trainStart = time.perf_counter()
model.fit(trainData, trainLabel, epochs=50, batch_size=batchSize)
model.evaluate(testData, testLabel, batch_size=batchSize)

trainEnd = time.perf_counter()
model.save(str(outputFolderName.joinpath("matrix_model.keras")))
keras.utils.plot_model(model, str(outputFolderName.joinpath("architecture.png")),
                       show_shapes=True, dpi=256)
print(f"Training time: {trainEnd - trainStart}")