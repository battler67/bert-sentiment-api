from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load model once
sentiment_pipeline = pipeline("sentiment-analysis")

@app.route('/')
def home():
    return "BERT Sentiment API is running!"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        result = sentiment_pipeline(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
