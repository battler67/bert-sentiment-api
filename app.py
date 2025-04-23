from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

# Load the sentiment model once to avoid reloading it for every request
sentiment_pipeline = pipeline("sentiment-analysis")

@app.route('/')
def home():
    return "BERT Sentiment API is running!"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No 'text' key provided in the request"}), 400
        text = data["text"]
        result = sentiment_pipeline(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure your app listens on the $PORT environment variable for Render
    port = int(os.getenv("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host='0.0.0.0', port=port)  # Use the dynamic $PORT environment variable
