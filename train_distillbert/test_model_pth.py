from transformers import DistilBertForSequenceClassification, DistilBertConfig
import torch
import os
from torch.serialization import add_safe_globals

# Add DistilBertConfig to safe globals
add_safe_globals([DistilBertConfig])

# Define the actual intent labels
INTENT_LABELS = [
    'EMI', 'COD', 'ORTHO_FEATURES', 'ERGO_FEATURES', 'COMPARISON', 
    'WARRANTY', '100_NIGHT_TRIAL_OFFER', 'SIZE_CUSTOMIZATION', 
    'WHAT_SIZE_TO_ORDER', 'LEAD_GEN', 'CHECK_PINCODE', 'DISTRIBUTORS', 
    'MATTRESS_COST', 'PRODUCT_VARIANTS', 'ABOUT_SOF_MATTRESS', 
    'DELAY_IN_DELIVERY', 'ORDER_STATUS', 'RETURN_EXCHANGE', 
    'CANCEL_ORDER', 'PILLOWS', 'OFFERS'
]

def load_model(model_path='./distilbert_model_trained/model.pth'):
    """
    Load the trained model and all necessary components from PyTorch format
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    # Load the checkpoint with weights_only=False since we need the config
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Initialize model with saved config
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        config=checkpoint['config']
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model, checkpoint['tokenizer'], checkpoint['label2id']

def predict_intent(model, tokenizer, text, label2id, top_k=10):
    """
    Predict the intent for a given text and return top k predictions with their probabilities
    """
    # Prepare the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get top k predictions
    values, indices = torch.topk(predictions[0], top_k)
    
    # Create list of predictions with their probabilities
    results = []
    for value, idx in zip(values, indices):
        prob = value.item()
        # Get the actual intent label directly from the index
        actual_label = INTENT_LABELS[idx.item()]
        label = f"LABEL_{idx.item()}"
        results.append((label, actual_label, prob))
    
    return results

def main():
    try:
        # Load the model and all components
        model, tokenizer, label2id = load_model()
        
        # Test with some example sentences
        test_sentences = [
            "What is the price of your mattress?",
            "Do you offer free delivery?",
            "What are your store hours?",
            "Can I return the mattress if I don't like it?"
        ]
        
        print("\nTesting the model with example sentences:")
        print("-" * 70)
        
        for sentence in test_sentences:
            predictions = predict_intent(model, tokenizer, sentence, label2id)
            print(f"\nInput: {sentence}")
            print("\nTop 10 Predictions:")
            print("-" * 70)
            print(f"{'Label Number':<15} {'Intent':<25} {'Confidence':<10}")
            print("-" * 70)
            for label, actual_label, confidence in predictions:
                print(f"{label:<15} {actual_label:<25} {confidence:.2%}")
            print("-" * 70)
        
       
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease make sure you have:")
        print("1. Trained the model first using DistilBERT.py")
        print("2. The model.pth file exists in the distilbert_model_trained directory")

if __name__ == "__main__":
    main() 