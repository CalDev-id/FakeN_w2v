import numpy as np
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, GlobalMaxPooling1D, Dense
from gensim.models import Word2Vec
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import tensorflow as tf
import numpy as np
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# Function for text preprocessing
def clean_text(text):
    # Lowercase and remove unwanted characters
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    text = stopword_remover.remove(text)
    return text

# Load and preprocess dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['text'] = data['Headline'] + " " + data['text']
    data['text'] = data['text'].apply(clean_text)
    return data

#load mendaley
# train_data = load_data("datasets/Mendaley/train.csv")
# val_data = load_data("datasets/Mendaley/validation.csv")
# test_data = load_data("datasets/Mendaley/test.csv")

# Load dataset tbh
train_data = load_data("datasets/TBH/train.csv")
val_data = load_data("datasets/TBH/validation.csv")
test_data = load_data("datasets/TBH/test.csv")

# # Load dataset github
# train_data = load_data("datasets/Github/train.csv")
# val_data = load_data("datasets/Github/validation.csv")
# test_data = load_data("datasets/Github/test.csv")

# Prepare tokenizer and encode labels
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['text'])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['text']), maxlen=128)
X_val = pad_sequences(tokenizer.texts_to_sequences(val_data['text']), maxlen=128)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['text']), maxlen=128)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['label'])
y_val = label_encoder.transform(val_data['label'])
y_test = label_encoder.transform(test_data['label'])

# Create Word2Vec embeddings
word2vec = Word2Vec(sentences=[text.split() for text in train_data['text']], vector_size=50, window=5, min_count=1, workers=4)
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 50))
for word, i in tokenizer.word_index.items():
    if word in word2vec.wv:
        embedding_matrix[i] = word2vec.wv[word]

# Build the Hybrid CNN-BiLSTM model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, weights=[embedding_matrix], input_length=128, trainable=False),
    Conv1D(32, kernel_size=3, activation='relu'),
    Dropout(0.5),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# Evaluate the model
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    y_pred_labels = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, average='weighted')
    recall = recall_score(y_true, y_pred_labels, average='weighted')
    f1 = f1_score(y_true, y_pred_labels, average='weighted')
    return acc, precision, recall, f1

acc, precision, recall, f1 = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
