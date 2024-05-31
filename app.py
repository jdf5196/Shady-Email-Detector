import os
from flask import Flask, request, jsonify, render_template

from model.preprocessing import preprocess_text
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

model = joblib.load(os.path.join(MODEL_DIR, 'phishing_model.pkl'))
vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data['emailText']
    processed_text = preprocess_text(email_text)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)[0]
    return jsonify({'shady': bool(prediction)})