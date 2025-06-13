from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertConfig
from datasets import Dataset
import numpy as np
import pandas as pd
import torch
import json
import os

def train_distilbert(train_data, model_name='distilbert-base-uncased', output_dir='./distilbert_model_trained'):
    """
    Train a DistilBERT model for sequence classification using only training data.
    Args:
        train_data (dict): Training data with 'sentence' and 'label' keys.
        model_name (str): Pretrained DistilBERT model name.
        output_dir (str): Directory to save the trained model.
    Returns:
        model (DistilBertForSequenceClassification): The trained model.
        tokenizer (DistilBertTokenizerFast): The tokenizer used.
    """
    # Calculate number of unique labels
    num_labels = len(set(train_data['label']))
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    
    # Initialize config with num_labels
    config = DistilBertConfig.from_pretrained(model_name, num_labels=num_labels)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, config=config)

    def tokenize_function(batch):
        return tokenizer(batch['sentence'], padding=True, truncation=True)

    train_dataset = Dataset.from_dict(train_data).map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        save_strategy="epoch",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save only the PyTorch model file with all necessary components
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer,
        'label2id': {label: idx for idx, label in enumerate(set(train_data['label']))}
    }, f"{output_dir}/model.pth")

    return model, tokenizer

def main():
    np.random.seed(42)

    train_df = pd.read_csv("../data/sofmattress_train.csv")

    if not {'sentence', 'label'}.issubset(train_df.columns):
        raise ValueError("CSV must contain 'sentence' and 'label' columns.")

    # Encode string labels to integers
    label2id = {label: idx for idx, label in enumerate(train_df['label'].unique())}
    train_df['label'] = train_df['label'].map(label2id)

    train_data = {
        'sentence': train_df['sentence'].tolist(),
        'label': train_df['label'].tolist()
    }

    model, tokenizer = train_distilbert(train_data)

    print("✅ Model training completed and saved as model.pth")

if __name__ == "__main__":
    main() 