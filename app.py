import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

sentiment_pipeline = pipeline("sentiment-analysis")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    result = sentiment_pipeline(text)
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # 👈 Use PORT env variable
    app.run(host='0.0.0.0', port=port)
