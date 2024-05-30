import os
from flask import Flask, request, jsonify, render_template
from scipy.sparse import hstack

from model.preprocessing import preprocess_text
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

model = joblib.load(os.path.join(MODEL_DIR, 'phishing_model.pkl'))
vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
sender_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_sender_email_vectorizer.pkl'))



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sender_email = data['senderEmail']
    email_text = data['emailText']
    processed_text = preprocess_text(email_text)
    text_features = vectorizer.transform([processed_text])
    sender_email_features = sender_vectorizer.transform([sender_email])
    features = hstack([text_features, sender_email_features])
    prediction = model.predict(features)[0]
    return jsonify({'phishing': bool(prediction)})