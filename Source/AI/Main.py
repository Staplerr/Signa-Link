import tensorflow as tf
import keras
from keras import layers
import pandas as pd
from pathlib import Path

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
dataset = pd.read_csv(Path(__file__).parent.joinpath("dataset.csv"))
print(dataset["Label"].convert_dtypes(convert_string=False).values)
print(dataset.drop(["Label"], axis=1).convert_dtypes(convert_string=False).values)
label = tf.convert_to_tensor(dataset["Label"].convert_dtypes().values)
data = tf.convert_to_tensor(dataset.drop(["Label"], axis=1).convert_dtypes().values, dtype=tf.float16)
print(data)

model = keras.models.Sequential([
    layers.Flatten(input_shape=(67, 3)),
    layers.Dense(len(labelList))
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#keras.utils.plot_model(model, to_file="model.png")
model.fit(data, label, epochs=5)