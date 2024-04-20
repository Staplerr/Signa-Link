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
]
ordered_sentences = [
    'ผมกินข้าวหมูกรอบ',
    'ผมกินข้าวขาหมู',
    'ผมกินข้าว',
    'ผมกินข้าวผัดปู',
]

# Tokenizing words
tokenizer = Tokenizer()
input_sequences = tokenizer.encode_list(sentences)
output_sequences = tokenizer.encode_list(ordered_sentences)

# Padding sequences
max_input_length = max(len(seq) for seq in input_sequences)
max_output_length = max(len(seq) for seq in output_sequences)

padded_input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    input_sequences, maxlen=max_input_length, padding='post'
)
padded_output_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    output_sequences, maxlen=max_output_length, padding='post'
)

# Prepare target sequences (shifted by one position)
target_sequences = np.zeros_like(padded_output_sequences)
target_sequences[:, :-1] = padded_output_sequences[:, 1:]

# Model configuration
vocab_size = tokenizer.n_vocab
embedding_dim = 100

# Define the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_input_length),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(padded_input_sequences, target_sequences, epochs=100, verbose=2)

result = model.predict('ข้าวผัดปูผมกิน')
print(result)
