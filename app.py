import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertConfig
from torch.serialization import add_safe_globals
import os

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

@st.cache_resource
def load_model(model_path='./train/distilbert_model_trained/model.pth'):
    """
    Load the trained model and all necessary components from PyTorch format
    """
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None, None, None
    
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
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

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
    st.set_page_config(
        page_title="Intent Detection for SOF Mattress",
        page_icon="🛏️",
        layout="wide"
    )
    
    st.title("🛏️ Intent Detection")
    st.markdown("""
    This application helps identify the intent behind customer queries related to SOF Mattress products.
    Enter a query below to see the predicted intents and their confidence scores.
    """)
    
    # Load the model
    model, tokenizer, label2id = load_model()
    
    if model is None:
        st.error("""
        Please make sure you have:
        1. Trained the model first using DistilBERT.py
        2. The model.pth file exists in the train/distilbert_model_trained directory
        """)
        return
    
    # Create a container for input
    input_container = st.container()
    
    with input_container:
        # Create two columns for input and button
        input_col, button_col = st.columns([4, 1])
        
        with input_col:
            # Input text area
            user_input = st.text_area(
                "Enter your question:",
                placeholder="Example: What is the price of your mattress?",
                height=100,
                key="query_input"
            )
        
        with button_col:
            st.write("")  # Add some vertical space
            st.write("")  # Add some vertical space
            submit_button = st.button("Enter", use_container_width=True)
    
    # Process input when either Enter key is pressed or button is clicked
    if user_input and (submit_button or st.session_state.get('query_input')):
        # Get predictions
        predictions = predict_intent(model, tokenizer, user_input, label2id)
        
        # Display results in a nice format
        st.markdown("### Predictions")
        
        # Create a container for the results
        results_container = st.container()
        
        with results_container:
            # Create columns for the table header
            col1, col2, col3 = st.columns([1, 2, 1])
            col1.markdown("**Label**")
            col2.markdown("**Intent**")
            col3.markdown("**Confidence**")
            
            # Display each prediction
            for label, actual_label, confidence in predictions:
                col1, col2, col3 = st.columns([1, 2, 1])
                col1.markdown(label)
                col2.markdown(actual_label)
                col3.markdown(f"{confidence:.2%}")
                
                # Add a progress bar for confidence
                st.progress(confidence)
        
        # Display the top prediction more prominently
        top_label, top_intent, top_confidence = predictions[0]
        st.markdown("### Top Prediction")
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
            <h2 style='color: #1f77b4;'>{top_intent}</h2>
            <p style='font-size: 1.2em;'>Confidence: {top_confidence:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
