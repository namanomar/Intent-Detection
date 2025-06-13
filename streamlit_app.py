import streamlit as st
import requests
import json
from typing import List, Dict

# Configure the page
st.set_page_config(
    page_title="Intent Detection for SOF Mattress",
    page_icon="🛏️",
    layout="wide"
)

# Constants
API_URL = "http://localhost:7860/predict"  # FastAPI endpoint

def call_api(text: str, top_k: int = 10) -> Dict:
    """
    Call the FastAPI endpoint to get predictions
    """
    try:
        response = requests.post(
            API_URL,
            json={"text": text, "top_k": top_k},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def display_predictions(predictions: List[Dict], top_prediction: Dict):
    """
    Display predictions in a nice format
    """
    # Display top prediction prominently
    st.markdown("### Prediction Result")
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
        <h2 style='color: #1f77b4;'>{top_prediction['actual_label']}</h2>
        <p style='font-size: 1.2em;'>Confidence: {top_prediction['confidence']:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

    # Display prediction details
    col1, col2 = st.columns([2, 1])
    col1.markdown("**Intent**")
    col2.markdown("**Confidence**")
    
    col1, col2 = st.columns([2, 1])
    col1.markdown(top_prediction['actual_label'])
    col2.markdown(f"{top_prediction['confidence']:.2%}")
    
    # Add a progress bar for confidence
    st.progress(top_prediction['confidence'])

def main():
    st.title("🛏️ Intent Detection")
    st.markdown("""
    This application helps identify the intent behind customer queries related to SOF Mattress products.
    Enter a query below to see the predicted intent and its confidence score.
    """)

    # Initialize session state for storing the last submitted query
    if 'last_submitted' not in st.session_state:
        st.session_state.last_submitted = None

    # Create a form for input
    with st.form(key='query_form'):
        # Input text area
        user_input = st.text_area(
            "Enter your question:",
            placeholder="Example: What is the price of your mattress?",
            height=100,
            key='query_input'
        )
        
        # Submit button
        submit_button = st.form_submit_button(label='Analyze Query')

    # Handle form submission
    if submit_button or (user_input and user_input != st.session_state.last_submitted):
        st.session_state.last_submitted = user_input
        top_k = 1

        # Show a spinner while waiting for the API response
        with st.spinner("Analyzing your query..."):
            # Call the API
            result = call_api(user_input, top_k)
            
            if result:
                # Display the predictions
                display_predictions(result['predictions'], result['top_prediction'])
            else:
                st.error("Failed to get predictions. Please try again.")

if __name__ == "__main__":
    main() 