import numpy as np
import pandas as pd
import re
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from transformers import TFBertModel, BertTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)

set_seed(42)

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
train_data = load_data("datasets/Mendaley/train.csv")
val_data = load_data("datasets/Mendaley/validation.csv")
test_data = load_data("datasets/Mendaley/test.csv")

# Load dataset tbh
# train_data = load_data("datasets/TBH/train.csv")
# val_data = load_data("datasets/TBH/validation.csv")
# test_data = load_data("datasets/TBH/test.csv")

# # Load dataset github
# train_data = load_data("datasets/Github/train.csv")
# val_data = load_data("datasets/Github/validation.csv")
# test_data = load_data("datasets/Github/test.csv")

# Prepare BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize and pad sequences
def encode_texts(texts, tokenizer, max_len=128):
    encoded = tokenizer(
        list(texts),
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors="tf"
    )
    return encoded

X_train = encode_texts(train_data['text'], bert_tokenizer, max_len=128)
X_val = encode_texts(val_data['text'], bert_tokenizer, max_len=128)
X_test = encode_texts(test_data['text'], bert_tokenizer, max_len=128)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['label'])
y_val = label_encoder.transform(val_data['label'])
y_test = label_encoder.transform(test_data['label'])
y_train = np.eye(len(label_encoder.classes_))[y_train]
y_val = np.eye(len(label_encoder.classes_))[y_val]
y_test = np.eye(len(label_encoder.classes_))[y_test]

# Load pre-trained BERT model
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

# Build the BERT-based classification model
input_ids = Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(128,), dtype=tf.int32, name="attention_mask")

bert_output = bert_model(input_ids, attention_mask=attention_mask)[1]
dropout = Dropout(0.5)(bert_output)
dense = Dense(len(label_encoder.classes_), activation="softmax")(dropout)

model = Model(inputs=[input_ids, attention_mask], outputs=dense)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# Train the model
history = model.fit(
    x={"input_ids": X_train["input_ids"], "attention_mask": X_train["attention_mask"]},
    y=y_train,
    validation_data=(
        {"input_ids": X_val["input_ids"], "attention_mask": X_val["attention_mask"]},
        y_val
    ),
    epochs=10,
    batch_size=16
)

# Evaluate the model
def evaluate_model(model, X, y_true):
    y_pred = model.predict({"input_ids": X["input_ids"], "attention_mask": X["attention_mask"]})
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    acc = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
    return acc, precision, recall, f1

acc, precision, recall, f1 = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
