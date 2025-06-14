import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import json
import os
from typing import List, Dict, Tuple, Optional
import pickle
from datetime import datetime
import joblib
from train_cnn.train_cnn import TextCNN
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the intent labels
INTENT_LABELS = [
    'EMI', 'COD', 'ORTHO_FEATURES', 'ERGO_FEATURES', 'COMPARISON', 
    'WARRANTY', '100_NIGHT_TRIAL_OFFER', 'SIZE_CUSTOMIZATION', 
    'WHAT_SIZE_TO_ORDER', 'LEAD_GEN', 'CHECK_PINCODE', 'DISTRIBUTORS', 
    'MATTRESS_COST', 'PRODUCT_VARIANTS', 'ABOUT_SOF_MATTRESS', 
    'DELAY_IN_DELIVERY', 'ORDER_STATUS', 'RETURN_EXCHANGE', 
    'CANCEL_ORDER', 'PILLOWS', 'OFFERS'
]

def generate_test_data() -> Tuple[List[str], List[str]]:
    """Generate test data for different intents"""
    
    
    texts, labels = zip(*test_cases)
    return list(texts), list(labels)

class ModelTester:
    def __init__(self, model_type: str, model_path: str):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> bool:
        """Load the model and required components"""
        try:
            if self.model_type == 'distilbert':
                return self._load_distilbert()
            elif self.model_type == 'svm':
                return self._load_svm()
            elif self.model_type == 'cnn':
                return self._load_cnn()
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return False
        except Exception as e:
            logger.error(f"Error loading {self.model_type} model: {str(e)}")
            return False
    
    def _load_distilbert(self) -> bool:
        """Load DistilBERT model and components"""
        try:
            from torch.serialization import add_safe_globals
            add_safe_globals([DistilBertConfig])
            
            # Load the checkpoint with weights_only=False since we need the config
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Initialize model with saved config
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                config=checkpoint['config']
            )
            
            # Load the state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer and label mapping
            self.tokenizer = checkpoint['tokenizer']
            self.label2id = checkpoint['label2id']
            
            # Create label encoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(INTENT_LABELS)
            
            logger.info(f"Successfully loaded DistilBERT model from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading DistilBERT model: {str(e)}")
            return False
    
    def _load_svm(self) -> bool:
        """Load SVM model and components"""
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load vectorizer
            vectorizer_path = os.path.join(os.path.dirname(self.model_path), 'tfidf_vectorizer.pkl')
            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load metadata
            metadata_path = os.path.join(os.path.dirname(self.model_path), 'model_metadata.json')
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create label encoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(metadata['training_data']['classes'])
            
            logger.info(f"Successfully loaded SVM model from {self.model_path}")
            logger.info(f"Model parameters: {metadata['model_parameters']}")
            return True
        except Exception as e:
            logger.error(f"Error loading SVM model: {str(e)}")
            return False
    
    def _load_cnn(self) -> bool:
        """Load CNN model and components"""
        try:
            # Load model data
            model_data = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model
            self.model = TextCNN(**model_data['model_params'])
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            tokenizer_path = os.path.join(os.path.dirname(self.model_path), 'tokenizer.pkl')
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load label encoder
            label_encoder_path = os.path.join(os.path.dirname(self.model_path), 'label_encoder.pkl')
            if not os.path.exists(label_encoder_path):
                raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}")
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            logger.info(f"Successfully loaded CNN model from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading CNN model: {str(e)}")
            return False
    
    def predict(self, texts: List[str]) -> Tuple[List[int], List[float], Optional[List[List[float]]]]:
        """Make predictions for the given texts"""
        if not self.model:
            logger.error("Model not loaded")
            return [], [], None
        
        try:
            if self.model_type == 'distilbert':
                return self._predict_distilbert(texts)
            elif self.model_type == 'svm':
                return self._predict_svm(texts)
            elif self.model_type == 'cnn':
                return self._predict_cnn(texts)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return [], [], None
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return [], [], None
    
    def _predict_distilbert(self, texts: List[str]) -> Tuple[List[int], List[float], List[List[float]]]:
        """Make predictions using DistilBERT model"""
        predictions = []
        confidences = []
        all_probs = []
        
        for text in texts:
            try:
                # Prepare the text
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    pred = torch.argmax(outputs.logits, dim=1).item()
                    conf = float(probs[0][pred])
                    
                    predictions.append(pred)
                    confidences.append(conf)
                    all_probs.append(probs[0].cpu().tolist())
            except Exception as e:
                logger.error(f"Error predicting for text '{text}': {str(e)}")
                predictions.append(0)
                confidences.append(0.0)
                all_probs.append([0.0] * len(self.label_encoder.classes_))
        
        return predictions, confidences, all_probs
    
    def _predict_svm(self, texts: List[str]) -> Tuple[List[int], List[float], Optional[List[List[float]]]]:
        """Make predictions using SVM model"""
        try:
            # Preprocess texts (lowercase and strip)
            texts = [text.lower().strip() for text in texts]
            
            # Transform texts using the vectorizer
            X_test = self.vectorizer.transform(texts)
            
            # Get predictions
            predictions = self.model.predict(X_test)
            
            # Get probabilities (SVM was trained with probability=True)
            probs = self.model.predict_proba(X_test)
            
            # Ensure predictions and probabilities are valid
            if len(predictions) != len(texts):
                raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match number of texts ({len(texts)})")
            
            if len(probs) != len(texts):
                raise ValueError(f"Number of probability arrays ({len(probs)}) doesn't match number of texts ({len(texts)})")
            
            # Calculate confidences and ensure they're valid
            confidences = []
            for prob in probs:
                if not np.isfinite(prob).all():
                    logger.warning("Found non-finite values in probabilities, using 0.0")
                    confidences.append(0.0)
                else:
                    confidences.append(float(max(prob)))
            
            # Log prediction statistics
            unique_preds = np.unique(predictions, return_counts=True)
            logger.info(f"Prediction distribution: {dict(zip(unique_preds[0], unique_preds[1]))}")
            logger.info(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            
            return predictions.tolist(), confidences, probs.tolist()
        except Exception as e:
            logger.error(f"Error in SVM prediction: {str(e)}")
            return [], [], None
    
    def _predict_cnn(self, texts: List[str]) -> Tuple[List[int], List[float], List[List[float]]]:
        """Make predictions using CNN model"""
        predictions = []
        confidences = []
        all_probs = []
        
        for text in texts:
            try:
                inputs = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
                    probs = torch.softmax(outputs, dim=1)
                    pred = torch.argmax(outputs, dim=1).item()
                    conf = float(probs[0][pred])
                    
                    predictions.append(pred)
                    confidences.append(conf)
                    all_probs.append(probs[0].cpu().tolist())
            except Exception as e:
                logger.error(f"Error predicting for text '{text}': {str(e)}")
                predictions.append(0)
                confidences.append(0.0)
                all_probs.append([0.0] * len(self.label_encoder.classes_))
        
        return predictions, confidences, all_probs

def calculate_metrics(true_labels: List[str], predictions: List[int], label_encoder: LabelEncoder) -> Dict:
    """Calculate performance metrics"""
    try:
        # Convert true labels to indices
        true_indices = label_encoder.transform(true_labels)
        
        # Ensure predictions are numpy array
        predictions = np.array(predictions)
        
        # Calculate basic metrics with zero_division=0
        accuracy = accuracy_score(true_indices, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_indices, predictions, 
            average='weighted',
            zero_division=0
        )
        
        # Calculate per-class metrics with zero_division=0
        class_report = classification_report(
            true_indices, predictions,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'per_class_metrics': class_report
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'per_class_metrics': {}
        }

def test_model(model_type: str, texts: List[str], true_labels: List[str], model_path: str) -> Dict:
    """Test a specific model"""
    try:
        tester = ModelTester(model_type, model_path)
        if not tester.load_model():
            return {
                'error': f"Failed to load {model_type} model",
                'accuracy': 0.0,
                'metrics': {},
                'detailed_results': []
            }
        
        predictions, confidences, all_probs = tester.predict(texts)
        if not predictions:
            return {
                'error': f"Failed to make predictions with {model_type} model",
                'accuracy': 0.0,
                'metrics': {},
                'detailed_results': []
            }
        
        # Log prediction statistics
        logger.info(f"\n{model_type.upper()} Model Statistics:")
        logger.info(f"Number of predictions: {len(predictions)}")
        logger.info(f"Number of unique predictions: {len(set(predictions))}")
        logger.info(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        
        # Calculate metrics
        metrics = calculate_metrics(true_labels, predictions, tester.label_encoder)
        
        # Prepare detailed results
        detailed_results = []
        for text, true_label, pred_idx, conf in zip(texts, true_labels, predictions, confidences):
            try:
                pred_label = tester.label_encoder.inverse_transform([pred_idx])[0]
                true_idx = tester.label_encoder.transform([true_label])[0]
                detailed_results.append({
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'is_correct': bool(pred_idx == true_idx),
                    'confidence': conf
                })
            except Exception as e:
                logger.error(f"Error processing result for text '{text}': {str(e)}")
                detailed_results.append({
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': 'ERROR',
                    'is_correct': False,
                    'confidence': 0.0
                })
        
        return {
            'accuracy': metrics['accuracy'],
            'metrics': metrics,
            'detailed_results': detailed_results,
            'has_probabilities': all_probs is not None
        }
    except Exception as e:
        logger.error(f"Error testing {model_type} model: {str(e)}")
        return {
            'error': str(e),
            'accuracy': 0.0,
            'metrics': {},
            'detailed_results': []
        }

def save_comparison_results(results: dict, output_dir: str = '.') -> str:
    """Save comparison results to a JSON file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'model_comparison_{timestamp}.json')
        
        # Prepare summary
        summary = {
            'timestamp': timestamp,
            'overall_accuracy': {
                model: results[model]['accuracy']
                for model in results if model != 'test_cases'
            },
            'model_details': {
                model: {
                    'accuracy': results[model]['accuracy'],
                    'metrics': results[model]['metrics'],
                    'has_probabilities': results[model]['has_probabilities']
                }
                for model in results if model != 'test_cases'
            },
            'performance_metrics': {
                'best_model': max(
                    (model, results[model]['accuracy'])
                    for model in results if model != 'test_cases'
                )[0],
                'model_rankings': sorted(
                    [(model, results[model]['accuracy']) for model in results if model != 'test_cases'],
                    key=lambda x: x[1],
                    reverse=True
                )
            }
        }
        
        # Add detailed results
        results['summary'] = summary
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error saving comparison results: {str(e)}")
        return ""

def main():
    try:
        # Generate test cases
        test_cases = generate_test_data()
        texts = test_cases[0]
        true_labels = test_cases[1]
        
        logger.info(f"Generated {len(texts)} test cases")
        
        # Test each model
        model_paths = {
            'distilbert': './train_distillbert/distilbert_model_trained/model.pth',
            'svm': './train_svm/svm_model.pkl',
            'cnn': './train_cnn/cnn_model.pth'
        }
        
        results = {}
        for model_type, model_path in model_paths.items():
            logger.info(f"\nTesting {model_type} model...")
            results[model_type] = test_model(model_type, texts, true_labels, model_path)
        
        # Add test cases to results
        results['test_cases'] = {
            'texts': texts,
            'labels': true_labels
        }
        
        # Save comparison results
        output_file = save_comparison_results(results)
        
        # Print summary
        logger.info("\nTest Results Summary:")
        for model_type in model_paths:
            logger.info(f"{model_type.upper()} Accuracy: {results[model_type]['accuracy']:.4f}")
        logger.info(f"\nDetailed results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
    
    