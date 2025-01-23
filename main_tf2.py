import tensorflow as tf
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from gensim.models import Word2Vec
# from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Unduh stopwords jika belum ada
# import nltk
# nltk.download('stopwords')
# stop_words = set(stopwords.words('indonesian'))

# # Fungsi untuk membersihkan teks
# def clean_text(text):
#     text = text.lower()  # Ubah menjadi huruf kecil
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Hapus URL
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus karakter non-alphabet
#     text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi ekstra
#     text = ' '.join([word for word in text.split() if word not in stop_words])  # Stopword removal
#     return text

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# Membuat stopword remover untuk bahasa Indonesia
factory = StopWordRemoverFactory()
stop_words = set(factory.get_stop_words())

# Fungsi untuk membersihkan teks (diperbarui untuk bahasa Indonesia)
def clean_text(text):
    text = text.lower()  # Ubah menjadi huruf kecil
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Hapus URL
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus karakter non-alphabet
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi ekstra
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Stopword removal
    return text

# Fungsi untuk membaca dan memproses data
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df['text_combined'] = df['Headline'] + " " + df['text']
    df['text_cleaned'] = df['text_combined'].apply(clean_text)
    return df['text_cleaned'], df['label']

# # Load dataset tbh
# train_texts, train_labels = load_and_preprocess('datasets/TBH/train.csv')
# val_texts, val_labels = load_and_preprocess('datasets/TBH/validation.csv')
# test_texts, test_labels = load_and_preprocess('datasets/TBH/test.csv')

# Load dataset mendaley
# train_texts, train_labels = load_and_preprocess('datasets/Mendaley/train.csv')
# val_texts, val_labels = load_and_preprocess('datasets/Mendaley/validation.csv')
# test_texts, test_labels = load_and_preprocess('datasets/Mendaley/test.csv')

# # Load dataset github
train_texts, train_labels = load_and_preprocess('datasets/Github/train.csv')
val_texts, val_labels = load_and_preprocess('datasets/Github/validation.csv')
test_texts, test_labels = load_and_preprocess('datasets/Github/test.csv')

# Label encoding
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)

# Tokenizer dan padding
max_words = 20000
max_len = 128  # Panjang sequence

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')
val_padded = pad_sequences(val_sequences, maxlen=max_len, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')

# Word2Vec embedding
embedding_dim = 50
word2vec = Word2Vec(sentences=[text.split() for text in train_texts], vector_size=embedding_dim, window=5, min_count=1)

# Membuat embedding matrix
vocab_size = min(max_words, len(tokenizer.word_index) + 1)
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = word2vec.wv.get_vector(word) if word in word2vec.wv else None
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Model CNN sesuai dengan arsitektur
input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=vocab_size, 
                            output_dim=embedding_dim, 
                            weights=[embedding_matrix], 
                            input_length=max_len, 
                            trainable=False)(input_layer)
conv_layer = Conv1D(filters=32, kernel_size=2, activation='relu')(embedding_layer)
dropout_layer = Dropout(0.5)(conv_layer)
pooling_layer = GlobalMaxPooling1D()(dropout_layer)
output_layer = Dense(len(label_encoder.classes_), activation='softmax')(pooling_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary (sesuai tabel)
model.summary()

# Training model
history = model.fit(train_padded, 
                    train_labels, 
                    validation_data=(val_padded, val_labels), 
                    epochs=10, 
                    batch_size=32)

# Evaluasi model
y_pred = model.predict(test_padded)
y_pred_classes = np.argmax(y_pred, axis=1)

# Metrics
accuracy = accuracy_score(test_labels, y_pred_classes)
precision = precision_score(test_labels, y_pred_classes, average='weighted')
recall = recall_score(test_labels, y_pred_classes, average='weighted')
f1 = f1_score(test_labels, y_pred_classes, average='weighted')

print("Test Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
