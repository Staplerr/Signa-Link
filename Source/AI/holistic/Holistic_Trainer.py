import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM, Dense, Flatten, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

# Load data
parent_directory = Path(__file__).parent
with open(parent_directory / "Data/labels.json", 'r') as f:
    label_dict = json.load(f)
features = np.load(parent_directory / "Data/Features.npy")
labels = np.load(parent_directory / "Data/Labels.npy")
print(f"Features: {features.shape}\nLabels: {labels.shape}")

# Normalize data
#features = features / 255.0

# Prepare data
encoder = OneHotEncoder()
labels = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
print(X_train.shape)

# Config
tf.keras.mixed_precision.set_global_policy('float32')
BATCHSIZE = 256

# Define the model
model = Sequential()
model.add(Input(shape=(10, 543, 3)))  # Add Input layer
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(256))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(len(labels[0]), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
output_dir = parent_directory / "Matrix model"
output_dir.mkdir(exist_ok=True)
checkpoint = ModelCheckpoint(output_dir / 'best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=BATCHSIZE, validation_data=(X_test, y_test),
                    callbacks=[checkpoint, early_stopping])

# Model summary
model.summary()

# Save the final model
model.save(output_dir / 'final_model.keras')
