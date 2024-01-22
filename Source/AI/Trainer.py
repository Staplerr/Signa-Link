import tensorflow as tf
import keras
from keras import layers
import pandas as pd
from pathlib import Path
import numpy as np
import ast
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, EarlyStopping

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

def preprocess_data():
    dataset = pd.read_excel(Path(__file__).parent.joinpath("dataset.xlsx"))
    label = tf.convert_to_tensor(dataset["Label"].values, dtype=tf.int8)
    stringData = dataset.drop(["Label"], axis=1)
    columnNames = [f'{i}' for i in range(len(stringData.columns))]
    data = pd.DataFrame(columns=columnNames)
    for row in stringData.values:
        rowData = [0] * len(stringData.columns)
        i = 0
        for matrix in row:
            if type(matrix) == str:
                matrix = ast.literal_eval(matrix)
            rowData[i] = matrix
            i += 1
        data.loc[len(data)] = rowData
    data = np.concatenate(data.to_numpy().flatten())
    data = data.reshape((-1, 201, 1))
    data = tf.convert_to_tensor(data, dtype=tf.float16)
    print(data.shape)
    return data, label

def create_model():
    model = keras.models.Sequential([
        layers.Flatten(input_shape=(201, 1)),
        layers.Dense(256, "relu"),
        layers.Dense(128, "relu"),
        layers.Dropout(0.3),
        layers.Dense(len(labelList))
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train(model, data, label, epochs, batch_size, callbacks=None, verbose=1):
    model.fit(data, label, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)

def save_model(model, number=1):
    model.save(fr"C:\Users\max\OneDrive\Desktop\Signa-Link-1\Source\AI\Model\{str(number)}.keras")
    print("\n\nsaved model")

def load_model(number=1):
    model = keras.models.load_model(fr"C:\Users\max\OneDrive\Desktop\Signa-Link-1\Source\AI\Model\{str(number)}.keras")
    return model


data, label = preprocess_data()
model = create_model()
train(model, data, label, epochs=100, batch_size=256, verbose=1)
print(model.summary())
