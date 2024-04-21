import numpy as np
from TOKENIZER import Tokenizer, separate_word
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding


sentences = [
    'ผมข้าวหมูกรอบกิน',
    'ผมข้าวขาหมูกิน',
    'ผมข้าวกิน',
    'ข้าวผัดปูผมกิน',
    'ข้าวข้าวผมข้าวผัดปูผมกินกินผม',
]
ordered_sentences = [
    'ผมกินข้าวหมูกรอบ',
    'ผมกินข้าวขาหมู',
    'ผมกินข้าว',
    'ผมกินข้าวผัดปู',
    'ผมกินข้าวผัดปู'
]

# Tokenizing words
tokenizer = Tokenizer()
input_sequences = tokenizer.encode_list(sentences)
output_sequences = tokenizer.encode_list(ordered_sentences)

# Don't uncomment. It broke everything.
'''# Model configuration
vocab_size = tokenizer.n_vocab
embedding_dim = 100

# Padding sequences
max_input_length = max(len(seq) for seq in input_sequences)
max_output_length = max(len(seq) for seq in output_sequences)

# Pad input sequences
padded_input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    input_sequences, maxlen=max_input_length, padding='post'
)

# Pad output sequences
padded_output_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    output_sequences, maxlen=max_output_length, padding='post'
)

# Prepare target sequences
target_sequences = np.zeros((len(output_sequences), max_output_length, vocab_size), dtype=np.float32)

# Assign the one-hot encoded vectors to target_sequences
for i, sequence in enumerate(output_sequences):
    for j, token_index in enumerate(sequence):
        target_sequences[i, j, token_index] = 1.0

# Define the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_input_length),
    LSTM(128, return_sequences=True),  # Return sequences for sequence-to-sequence model
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(padded_input_sequences, target_sequences, epochs=100, verbose=2)

# Predict method input
input_text = 'ข้าวผัดปูผมกิน'
input_sequence = tokenizer.encode(input_text)
padded_input_sequence = tf.keras.preprocessing.sequence.pad_sequences(
    [input_sequence], maxlen=max_input_length, padding='post'
)

# Make prediction
result = model.predict(padded_input_sequence)
print(result)'''