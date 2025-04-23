from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

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
        # Get the JSON data from the request
        data = request.get_json()
        
        # Ensure 'text' key exists in the request
        if not data or "text" not in data:
            return jsonify({"error": "No 'text' key provided in the request"}), 400
        
        text = data["text"]
        
        # Perform sentiment analysis using BERT model
        result = sentiment_pipeline(text)
        
        # Return the sentiment analysis result
        return jsonify(result)
    
    except Exception as e:
        # In case of errors, return the error message
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)  # Make sure app runs correctly in Render's environment
