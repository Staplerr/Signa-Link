import tensorflow as tf
import keras
from keras import layers
from tensorflow import nn
import pandas as pd
from pathlib import Path
import numpy as np
import ast

keras.mixed_precision.set_global_policy(keras.mixed_precision.Policy('float32'))
batchSize = 128
parentDirectory = Path(__file__).parent
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
def convertDataset(dataset):
    label = tf.convert_to_tensor(dataset["Label"].values, dtype=tf.int8)
    stringData = dataset.drop(["Label"], axis=1)
    columnNames = [f'{i}' for i in range(len(stringData.columns))]
    data = pd.DataFrame(columns=columnNames)
    for row in stringData.values: #pandas decided to convert all matrix to string when save the dataset as xlsx
        rowData = [0] * len(stringData.columns)
        i = 0
        for matrix in row:
            if type(matrix) == str: #check for 0 which pandas doesn't turn into string
                matrix = ast.literal_eval(matrix)
            rowData[i] = matrix
            i += 1
        data.loc[len(data)] = rowData
    data = np.concatenate(data.to_numpy().flatten())
    data = data.reshape((-1, 67*3, 1))
    data = tf.convert_to_tensor(data, dtype=tf.float16)
    return label, data
dataset = pd.read_excel(parentDirectory.joinpath("dataset1.xlsx"))
label, data = convertDataset(dataset)
del dataset 

model = keras.models.Sequential([
    layers.Flatten(input_shape=(201, 1)),
    layers.Dense(256, activation=nn.relu),
    layers.Dense(512, activation=nn.relu),
    layers.Dense(256, activation=nn.relu),
    layers.Dense(128, activation=nn.leaky_relu),
    layers.Dropout(0.3),
    layers.Dense(len(labelList), activation=nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, label, epochs=50, batch_size=batchSize)
model.evaluate(data, label, batch_size=batchSize)
model.save(str(parentDirectory.joinpath("Matrix model smol")))