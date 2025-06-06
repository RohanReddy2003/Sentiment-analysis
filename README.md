# Social Media Sentiment Analyzer with Multimodal Fusion

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

A hybrid sentiment analysis system combining **LSTM text analysis** and **DeepFace emotion detection** for comprehensive social media content evaluation.

## Key Innovations 🚀
- **Bidirectional LSTM** trained on 165K+ social media samples (90.1% accuracy)
- **Real-time facial sentiment analysis** using DeepFace's VGG-Face model
- **Confidence-weighted fusion algorithm** for multimodal decision making
- **Production-ready Flask API** with React frontend

## Technical Architecture 🧠
```mermaid
flowchart TD
    A[User Upload] --> B{Image&Text}
    B -->|Image| C[EasyOCR]
    C -->|Text| D[Text pre-processing & tokenization]
    D --> E[Bi-LSTM model]
    E --> F[Text Sentiment]
    C -->|Face| G[DeepFace Emotion Analysis]
    G --> H[Facial sentiment]
    F & H --> I[Fusion Algorithm]
    I --> J[(Final Sentiment)]
```
## Installation

# Clone repository
git clone https://github.com/yourusername/sentiment-fusion.git
cd sentiment-fusion

# Create Virtual Environment
python -m venv .venv

# Activate venv
.venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Download models
wget https://yourmodelhost/sentiment_analyzer.pkl -P models/

# Run Flask app
python app.py
