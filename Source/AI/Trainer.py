import tensorflow as tf
import keras
from keras import layers
from tensorflow import nn
import pandas as pd
from pathlib import Path
import numpy as np
import ast
import time

parentDirectory = Path(__file__).parent
datasetType = "smol"
keras.mixed_precision.set_global_policy(keras.mixed_precision.Policy('mixed_float16'))
batchSize = 512
labelList = {"กรอบ": 0,
             "กระเพรา": 1,
             "ขา": 2,
             "ข้าว": 3,
             "ไข่": 4,
             "คะน้า": 5,
             "เค็ม": 6,
             "โจ๊ก": 7,
             "แดง": 8,
             "ต้ม": 9,
             "แตงโม": 10,
             "น้ำพริกเผา": 11,
             "บะหมี่": 12,
             "เปรี้ยว": 13,
             "ผัด": 14,
             "ฝรั่ง": 15,
             "พริกแกง": 16,
             "มะม่วง": 17,
             "ม้า": 18,
             "มาม่า": 19,
             "ลูกชิ้นปลา": 20,
             "เลือด": 21,
             "สับ": 22,
             "เส้นเล็ก": 23,
             "เส้นใหญ่": 24,
             "หมู": 25,
             "หวาน": 26,
             "องุ่น": 27,
             "แอปเปิ้ล": 28}

def preprocessData(dataset):
    label = dataset["Label"].values
    #label = tf.convert_to_tensor(label, dtype=tf.int8) #Convert to tensor

    stringData = dataset.drop(["Label"], axis=1)
    columnNames = [f'{i}' for i in range(len(stringData.columns))]
    data = pd.DataFrame(columns=columnNames)
    for row in stringData.values: #pandas decided to convert all matrix to string when save the dataset as xlsx
        rowData = [0] * len(stringData.columns)
        i = 0
        for matrix in row:
            rowData[i] = ast.literal_eval(matrix) #turn string into matrix
            i += 1
        data.loc[len(data)] = rowData
    #convert dataframe to ndarray
    #convert the 3D ndarray into 2D ndarray or converting [[[x,y,z],[x,y,z],...],[[x,y,z],[x,y,z],...],...] to [[x,y,z],[x,y,z],[x,y,z],...] with .flatten() function
    #convert the 2D ndarray into 1D ndarray or converting [[x,y,z],[x,y,z],[x,y,z],...] to [x,y,z,x,y,z,x,y,z,...] with np.concatenate() function
    data = np.concatenate(data.to_numpy().flatten())
    data = data.reshape((-1, len(stringData.columns)*3, 1)) #sperate ndarray into multiple one corresponding to its label
    #data = tf.convert_to_tensor(data, dtype=tf.float16) #Convert to tensor
    return data, label

def splitData(data, label, ratio):
    #trainData, testData = np.split(data, int(ratio * len(data)))
    #trainLabel, testLabel = np.split(label, int(ratio * len(label)))
    trainData = data[0:int(ratio * len(data))]
    testData = data[int(ratio * len(data)):-1]

    trainLabel = label[0:int(ratio * len(label))]
    testLabel = label[int(ratio * len(label)):-1]
    return trainData, testData, trainLabel, testLabel

#Preparing dataset
dataset = pd.read_excel(parentDirectory.joinpath(f"{datasetType} dataset.xlsx"))
loadStart = time.perf_counter()
data, label = preprocessData(dataset)
trainData, testData, trainLabel, testLabel = splitData(data, label, 0.8)
loadFinish = time.perf_counter()
print(f"Dataset load time: {loadFinish - loadStart}")
print(len(data))
print(len(trainData))
del dataset

model = keras.models.Sequential([
    layers.Flatten(input_shape=(201, 1)),
    layers.Dropout(0.5),
    layers.Dense(256, activation=nn.relu),
    layers.Dropout(0.2),
    layers.Dense(256, activation=nn.relu),
    layers.Dense(len(labelList), activation=nn.softmax)
])
model.compile(optimizer=keras.optimizers.Adam,
              loss=keras.losses.SparseCategoricalCrossentropy,
              metrics=['accuracy'])
trainStart = time.perf_counter()
model.fit(trainData, trainLabel, epochs=50, batch_size=batchSize)
model.evaluate(testData, testLabel, batch_size=batchSize)
trainEnd = time.perf_counter()
model.save(str(parentDirectory.joinpath(f"Matrix model {datasetType}")))
print(f"Training time: {trainEnd - trainStart}")