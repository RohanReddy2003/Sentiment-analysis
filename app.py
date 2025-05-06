from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import cv2
import easyocr
from deepface import DeepFace

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
def load_models():
    # Load the combined pickle file containing tokenizer and LSTM model
    with open('models/sentiment_analyzer.pkl', 'rb') as f:
        analyzer = pickle.load(f)
    
    # Initialize OCR reader
    ocr_reader = easyocr.Reader(['en'])
    
    return analyzer, ocr_reader

analyzer, ocr_reader = load_models()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            results = analyze_image(filepath)
            
            return render_template('results.html', 
                                 image_path=filename,
                                 results=results)
    
    return render_template('index.html')

def analyze_image(image_path):
    # Extract text from image
    results = ocr_reader.readtext(image_path)
    extracted_text = " ".join([res[1] for res in results])
    
    # Predict text sentiment using LSTM from pickle
    tokenizer = analyzer['tokenizer']
    lstm_model = analyzer['model']
    
    seq = tokenizer.texts_to_sequences([extracted_text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    preds = lstm_model.predict(padded, verbose=0)
    text_label_idx = np.argmax(preds[0])
    text_conf = preds[0][text_label_idx]
    
    label_map = {0: "Negative", 1: "Happy", 2: "Neutral", 3: "Sad"}
    text_label = label_map[text_label_idx]
    
    # Facial sentiment prediction
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        
        face_label_map = {
            "angry": "Negative",
            "disgust": "Negative",
            "fear": "Negative",
            "sad": "Negative",
            "happy": "Positive",
            "surprise": "Positive",
            "neutral": "Neutral"
        }
        face_label = face_label_map.get(emotion.lower(), "Neutral")
        # Normalize to [0, 1]
        face_conf = analysis[0]['emotion'][emotion] / 100
    except Exception as e:
        print(f"DeepFace error: {e}")
        face_label = "Neutral"
        face_conf = 0.5
    
    # Fuse sentiments
    final_sentiment = fuse_sentiments(text_label, text_conf, face_label, face_conf)
    
    return {
        'text': extracted_text,
        'text_sentiment': text_label,
        'text_confidence': float(text_conf),
        'face_sentiment': face_label,
        'face_confidence': float(face_conf),
        'final_sentiment': final_sentiment
    }

def fuse_sentiments(text_label, text_conf, face_label, face_conf):
    sentiment_score = {"Negative": -1, "Positive": 1, "Neutral": 0,
                       "Happy": 1, "Sad": -1}
    
    text_score = sentiment_score.get(text_label, 0) * text_conf
    face_score = sentiment_score.get(face_label, 0) * face_conf
    
    combined_score = (text_score + face_score) / (text_conf + face_conf)
    
    if combined_score >= 0.5:
        return "Positive"
    elif combined_score <= -0.5:
        return "Negative"
    else:
        return "Neutral"

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = analyze_image(filepath)
        
        return jsonify(results)
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)