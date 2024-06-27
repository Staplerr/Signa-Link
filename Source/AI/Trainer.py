import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import Video_converter

# load data
labelDict = Video_converter.labelList
Data = np.load("Data.npy")
Label = np.load("Label.npy")
print(f"Features : {Data.shape}\nLabels : {Label.shape}")

# prepare data
encoder = OneHotEncoder()
Label = encoder.fit_transform(Label.reshape(-1, 1)).toarray()
X_train, X_test, y_train, y_test = train_test_split(Data ,Label ,
                                                    test_size=0.3, random_state = 42)

# config
tf.keras.mixed_precision.set_global_policy('float32') # floating type
BATCHSIZE = 256



# model
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(10, 42, 3)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(Label[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

history = model.fit(tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train),
                     epochs=100, batch_size=BATCHSIZE, validation_data=(tf.convert_to_tensor(X_test),
                                                                         tf.convert_to_tensor(y_test)),
                    callbacks=[checkpoint, early_stopping])

model.save('final_model.h5')

'''parentDirectory = Path(__file__).parent
configFilePath = parentDirectory.joinpath("config.cfg")
if not configFilePath.exists():
    raise Exception("No config file found")
configFile = configparser.RawConfigParser()
configFile.read(configFilePath)

#Change preference in config.cfg
policy = configFile['Options']['policy']
batchSize = int(configFile['Options']['batchSize'])

keras.mixed_precision.set_global_policy(policy)
modelName = parentDirectory.joinpath(f"Matrix model/matrix_model_{policy}")

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
    layers.InputLayer(input_shape=(2010,)),
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
model.save(modelName)
#keras.utils.plot_model(model, str(outputFolderName.joinpath("architecture.png")),
#                       show_shapes=True, dpi=256)
print(f"Training time: {trainEnd - trainStart}")'''