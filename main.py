import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
from transformers import AutoTokenizer

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, word2vec_model):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.word2vec_model = word2vec_model
        self.unk_vector = np.zeros(self.word2vec_model.vector_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['Headline'] + " " + row['text']
        tokens = self.tokenizer.tokenize(text)[:self.max_len]
        embeddings = [
            self.word2vec_model.wv[token] if token in self.word2vec_model.wv else self.unk_vector
            for token in tokens
        ]
        if len(embeddings) < self.max_len:
            embeddings += [self.unk_vector] * (self.max_len - len(embeddings))

        label = row['label']
        return torch.tensor(embeddings, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Model definition
class TextClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes, lr):
        super(TextClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(128, num_classes)
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Shape: (batch_size, input_dim, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        preds = torch.argmax(outputs, dim=1)
        return {'labels': labels.cpu(), 'preds': preds.cpu()}

    def test_epoch_end(self, outputs):
        all_labels = torch.cat([x['labels'] for x in outputs])
        all_preds = torch.cat([x['preds'] for x in outputs])
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        self.log_dict({'test_acc': acc, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1})

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Main script
if __name__ == '__main__':
    # Paths to datasets
    train_path = '/datasets/train.csv'
    val_path = '/datasets/validation.csv'
    test_path = '/datasets/test.csv'

    # Hyperparameters
    max_len = 50
    embedding_dim = 100
    lr = 1e-3
    num_classes = 3  # Update based on your dataset
    batch_size = 32

    # Load data
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    test_data = load_data(test_path)

    # Train Word2Vec model
    sentences = (train_data['Headline'] + " " + train_data['text']).apply(lambda x: x.split()).tolist()
    word2vec_model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset = TextDataset(train_data, tokenizer, max_len, word2vec_model)
    val_dataset = TextDataset(val_data, tokenizer, max_len, word2vec_model)
    test_dataset = TextDataset(test_data, tokenizer, max_len, word2vec_model)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    model = TextClassifier(input_dim=embedding_dim, num_classes=num_classes, lr=lr)

    # Trainer
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1 if torch.cuda.is_available() else None)

    # Train and validate
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model, test_loader)
