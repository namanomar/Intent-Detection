import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import json

# Define the intent labels
INTENT_LABELS = [
    'EMI', 'COD', 'ORTHO_FEATURES', 'ERGO_FEATURES', 'COMPARISON', 
    'WARRANTY', '100_NIGHT_TRIAL_OFFER', 'SIZE_CUSTOMIZATION', 
    'WHAT_SIZE_TO_ORDER', 'LEAD_GEN', 'CHECK_PINCODE', 'DISTRIBUTORS', 
    'MATTRESS_COST', 'PRODUCT_VARIANTS', 'ABOUT_SOF_MATTRESS', 
    'DELAY_IN_DELIVERY', 'ORDER_STATUS', 'RETURN_EXCHANGE', 
    'CANCEL_ORDER', 'PILLOWS', 'OFFERS'
]

def load_data(csv_path: str) -> pd.DataFrame:
    """Load data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return None

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess the data for training"""
    # Convert text to lowercase
    df['sentence'] = df['sentence'].str.lower()
    
    # Remove any leading/trailing whitespace
    df['sentence'] = df['sentence'].str.strip()
    
    # Split into features and labels
    X = df['sentence'].values
    y = df['label'].values
    
    return X, y

def train_svm_model(X_train, y_train, X_val, y_val):
    """Train SVM model with hyperparameter tuning"""
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),  # Use both unigrams and bigrams
        min_df=2,  # Minimum document frequency
        max_df=0.95  # Maximum document frequency
    )
    
    # Transform the training and validation data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    # Initialize SVM model
    svm = SVC(probability=True)
    
    # Perform grid search
    print("\nPerforming grid search for best parameters...")
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train_tfidf, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Print best parameters
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    
    # Evaluate on validation set
    y_pred = best_model.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return best_model, vectorizer

def save_model_and_metadata(model, vectorizer, metadata, output_dir: str):
    """Save the model, vectorizer, and metadata"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'svm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {vectorizer_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

def main():
    # Load data
    data_path = "../data/sofmattress_train.csv"
    df = load_data(data_path)
    if df is None:
        return
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model, vectorizer = train_svm_model(X_train, y_train, X_val, y_val)
    
    # Prepare metadata
    metadata = {
        "model_type": "SVM",
        "vectorizer": "TF-IDF",
        "features": {
            "max_features": 1000,
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95
        },
        "training_data": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "num_classes": len(INTENT_LABELS),
            "classes": INTENT_LABELS
        },
        "model_parameters": model.get_params()
    }
    
    # Save model and metadata
    save_model_and_metadata(model, vectorizer, metadata, ".")

if __name__ == "__main__":
    main() 