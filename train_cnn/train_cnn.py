import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import json
import os
from typing import List, Dict, Tuple
import pickle
from transformers import DistilBertTokenizer
from tqdm import tqdm

# Define the intent labels
INTENT_LABELS = [
    'EMI', 'COD', 'ORTHO_FEATURES', 'ERGO_FEATURES', 'COMPARISON', 
    'WARRANTY', '100_NIGHT_TRIAL_OFFER', 'SIZE_CUSTOMIZATION', 
    'WHAT_SIZE_TO_ORDER', 'LEAD_GEN', 'CHECK_PINCODE', 'DISTRIBUTORS', 
    'MATTRESS_COST', 'PRODUCT_VARIANTS', 'ABOUT_SOF_MATTRESS', 
    'DELAY_IN_DELIVERY', 'ORDER_STATUS', 'RETURN_EXCHANGE', 
    'CANCEL_ORDER', 'PILLOWS', 'OFFERS'
]

class TextDataset(Dataset):
    """Custom Dataset for text data"""
    def __init__(self, texts: List[str], labels: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Convert labels to indices
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(INTENT_LABELS)
        self.label_indices = self.label_encoder.transform(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label_indices[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class TextCNN(nn.Module):
    """CNN model for text classification"""
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, filter_sizes: List[int], 
                 output_dim: int, dropout: float, pad_idx: int):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, 
                     kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, emb_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, emb_dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device,
                n_epochs: int = 10) -> Dict:
    """Train the model"""
    best_val_loss = float('inf')
    best_model_state = None
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{n_epochs} [Train]'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{n_epochs} [Val]'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_steps += 1
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / val_steps
        val_accuracy = 100 * correct / total
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f'Epoch {epoch + 1}/{n_epochs}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
    
    return best_model_state, history

def save_model_and_metadata(model: nn.Module, tokenizer, label_encoder, history: Dict, 
                          model_params: Dict, output_dir: str):
    """Save the model and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'cnn_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': model_params
    }, model_path)
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save label encoder
    label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nModel and metadata saved to {output_dir}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_path = "../data/sofmattress_train.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['sentence'].values, df['label'].values,
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Model parameters
    model_params = {
        'vocab_size': tokenizer.vocab_size,
        'embedding_dim': 300,
        'n_filters': 100,
        'filter_sizes': [3, 4, 5],
        'output_dim': len(INTENT_LABELS),
        'dropout': 0.5,
        'pad_idx': tokenizer.pad_token_id
    }
    
    # Initialize model
    model = TextCNN(**model_params).to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    print("\nStarting training...")
    best_model_state, history = train_model(
        model, train_loader, val_loader, optimizer, criterion, device
    )
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    # Save model and metadata
    save_model_and_metadata(
        model, tokenizer, train_dataset.label_encoder, history, model_params, "."
    )

if __name__ == "__main__":
    main() 