import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPooling1D, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import tensorflow as tf
import numpy as np
import random

# random.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocessing function
def preprocess_text(text, stopword_remover):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    text = text.lower()
    # Remove stopwords
    text = stopword_remover.remove(text)
    return text

# Prepare the data
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

def prepare_data(file_path):
    data = load_data(file_path)
    data['text'] = data['Headline'] + ' ' + data['text']
    data['text'] = data['text'].apply(lambda x: preprocess_text(x, stopword_remover))
    return data['text'].values, data['label'].values



# Load dataset mendaley
# train_texts, train_labels = prepare_data('datasets/Mendaley/train.csv')
# val_texts, val_labels = prepare_data('datasets/Mendaley/validation.csv')
# test_texts, test_labels = prepare_data('datasets/Mendaley/test.csv')

# # Load dataset tbh
# train_texts, train_labels = prepare_data('datasets/TBH/train.csv')
# val_texts, val_labels = prepare_data('datasets/TBH/validation.csv')
# test_texts, test_labels = prepare_data('datasets/TBH/test.csv')

# # Load dataset github
train_texts, train_labels = prepare_data('datasets/Github/train.csv')
val_texts, val_labels = prepare_data('datasets/Github/validation.csv')
test_texts, test_labels = prepare_data('datasets/Github/test.csv')

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1

train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

max_length = 128
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Train Word2Vec model
w2v_model = Word2Vec(sentences=[text.split() for text in train_texts], vector_size=50, window=5, min_count=1, workers=4)

# Create embedding matrix
embedding_dim = 50
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# Build the BiLSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_padded, train_labels,
    validation_data=(val_padded, val_labels),
    epochs=10,
    batch_size=32
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_padded, test_labels)
test_preds = np.argmax(model.predict(test_padded), axis=-1)

# Calculate metrics
precision = precision_score(test_labels, test_preds, average='weighted')
recall = recall_score(test_labels, test_preds, average='weighted')
f1 = f1_score(test_labels, test_preds, average='weighted')

print(f"Test Accuracy: {test_acc}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1 Score: {f1}")
