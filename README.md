# Intent Detection for SOF Mattress

A machine learning-based intent detection system for customer queries related to SOF Mattress products. The system supports multiple models (SVM, DistilBERT, and CNN) for natural language understanding and provides a user-friendly interface through Streamlit.

<img src="./image.png"></img>

## Features

- Multiple model support:
  - SVM with TF-IDF vectorization
  - DistilBERT for advanced NLP
  - CNN for text classification
- Fast and accurate intent detection
- Real-time prediction with confidence scores
- User-friendly web interface
- RESTful API endpoint for integration
- Docker support for easy deployment
- Comprehensive model testing and evaluation

## Project Structure

```
.
├── api.py                 # FastAPI backend for intent detection
├── streamlit_app.py       # Streamlit frontend interface
├── test_model.py         # Model testing and evaluation script
├── requirements.txt       # Python package dependencies
├── Dockerfile            # Docker configuration for containerization
├── .dockerignore         # Docker build exclusions
├── train/                # Training related files
│   ├── train_svm/        # SVM model training
│   │   ├── svm_model.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   └── model_metadata.json
│   ├── train_distillbert/  # DistilBERT model training
│   │   └── model.pth
│   └── train_cnn/        # CNN model training
│       └── cnn_model.pth
└── README.md             # Project documentation
```

## Models

### 1. SVM Model
- Uses TF-IDF vectorization for text preprocessing
- Linear kernel with probability estimates
- Fast inference time
- Good for smaller datasets

### 2. DistilBERT Model
- Based on DistilBERT architecture
- Fine-tuned for intent classification
- Better performance on complex queries

### 3. CNN Model
- Custom CNN architecture for text classification
- Multiple filter sizes for feature extraction

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/namanomar/Intent-Detection
cd Intent-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Model Training

You can train any of the three models:

```bash
# Train SVM model
cd train_svm
python train_svm.py

# Train DistilBERT model
cd train_distillbert
python train.py

# Train CNN model
cd train_cnn
python train_cnn.py
```

4. Test Models

To evaluate and compare all models:
```bash
python test_model.py
```

5. Run the FastAPI backend:
```bash
python api.py
```

6. In a separate terminal, run the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

### Docker Setup

1. Build the Docker image:
```bash
docker build -t intent-detection-api .
```

2. Run the container:
```bash
docker run -p 7860:7860 intent-detection-api
```

## API Usage

The API provides a `/predict` endpoint that accepts POST requests with the following format:

```json
{
    "text": "What is the price of your mattress?",
    "top_k": 10
}
```

Response format:
```json
{
    "predictions": [
        {
            "label": "LABEL_12",
            "actual_label": "MATTRESS_COST",
            "confidence": 0.95
        },
        ...
    ],
    "top_prediction": {
        "label": "LABEL_12",
        "actual_label": "MATTRESS_COST",
        "confidence": 0.95
    }
}
```

## Model Testing

The `test_model.py` script provides comprehensive testing capabilities:

- Tests all models on a diverse set of queries
- Generates detailed performance metrics
- Creates comparison reports
- Supports custom test cases

Run tests with:
```bash
python test_model.py
```

## Performance Metrics

The system evaluates models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Supported Intents

The system can detect the following intents:
- EMI
- COD
- ORTHO_FEATURES
- ERGO_FEATURES
- COMPARISON
- WARRANTY
- 100_NIGHT_TRIAL_OFFER
- SIZE_CUSTOMIZATION
- WHAT_SIZE_TO_ORDER
- LEAD_GEN
- CHECK_PINCODE
- DISTRIBUTORS
- MATTRESS_COST
- PRODUCT_VARIANTS
- ABOUT_SOF_MATTRESS
- DELAY_IN_DELIVERY
- ORDER_STATUS
- RETURN_EXCHANGE
- CANCEL_ORDER
- PILLOWS
- OFFERS

## API Documentation

- Swagger UI: `http://localhost:7860/docs`

