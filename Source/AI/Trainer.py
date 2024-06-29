import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

# Load data
parentDirectory = Path(__file__).parent
f = open(f"{str(parentDirectory)}/Data/label.json",)
labelDict = json.load(f)
Features = np.load(f"{str(parentDirectory)}/Data/Features.npy")
Label = np.load(f"{str(parentDirectory)}/Data/Label.npy")
print(f"Features : {Features.shape}\nLabels : {Label.shape}")

# Prepare data
encoder = OneHotEncoder()
Label = encoder.fit_transform(Label.reshape(-1, 1)).toarray()
X_train, X_test, y_train, y_test = train_test_split(Features, Label, test_size=0.3, random_state=42)
print(X_train.shape)

# Config
tf.keras.mixed_precision.set_global_policy('float32')  # Set global policy for mixed precision
BATCHSIZE = 256

# Define the model
model = Sequential()
model.add(Conv2D(4, (3,3), data_format="channels_last", input_shape=(Features.shape[1], Features.shape[2], Features.shape[3])))
model.add(MaxPooling2D((2, 2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(256, activation='relu'))
model.add(Dense(len(Label[0]), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=BATCHSIZE, validation_data=(X_test, y_test),
                    callbacks=[checkpoint, early_stopping])

# Model summary
model.summary()

# Save the final model
model.save(f'{str(parentDirectory.joinpath("Matrix model"))}/final_model.keras')
