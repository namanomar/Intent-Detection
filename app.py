from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import DistilBertForSequenceClassification, DistilBertConfig
from torch.serialization import add_safe_globals
import os
from typing import List, Tuple

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

# Initialize FastAPI app
app = FastAPI(
    title="Intent Detection API",
    description="API for detecting intents in customer queries related to SOF Mattress products",
    version="1.0.0"
)

# Global variables for model, tokenizer, and label2id
model = None
tokenizer = None
label2id = None

class Query(BaseModel):
    text: str
    top_k: int = 10

class Prediction(BaseModel):
    label: str
    actual_label: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]
    top_prediction: Prediction

def load_model(model_path='./train/distilbert_model_trained/model.pth'):
    """
    Load the trained model and all necessary components from PyTorch format
    """
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail=f"Model file not found at {model_path}. Please train the model first.")
    
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def predict_intent(text: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
    """
    Predict the intent for a given text and return top k predictions with their probabilities
    """
    global model, tokenizer, label2id
    
    if model is None or tokenizer is None or label2id is None:
        model, tokenizer, label2id = load_model()
    
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

@app.on_event("startup")
async def startup_event():
    """
    Load the model when the application starts
    """
    global model, tokenizer, label2id
    model, tokenizer, label2id = load_model()

@app.post("/predict", response_model=PredictionResponse)
async def predict(query: Query):
    """
    Predict intents for the given text
    """
    predictions = predict_intent(query.text, query.top_k)
    
    # Convert predictions to response format
    prediction_list = [
        Prediction(label=label, actual_label=actual_label, confidence=confidence)
        for label, actual_label, confidence in predictions
    ]
    
    # Get top prediction
    top_prediction = prediction_list[0]
    
    return PredictionResponse(
        predictions=prediction_list,
        top_prediction=top_prediction
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
