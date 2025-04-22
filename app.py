from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
import torch

# Initialize FastAPI app
app = FastAPI()

# Load the BERT sentiment analysis model
# Note: This may take a few seconds during cold starts
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Define the data model for the input
class Review(BaseModel):
    review: str

# Root endpoint (for Render health check)
@app.get("/")
async def home():
    return {"message": "BERT Sentiment Analysis API is live!"}

# Endpoint to analyze sentiment
@app.post("/analyze")
async def analyze_sentiment(data: Review):
    try:
        result = sentiment_pipeline(data.review)[0]
        label = result["label"]
        score = result["score"]

        # Map star labels to custom sentiment labels
        if label in ["1 star", "2 stars"]:
            sentiment = "Negative"
        elif label == "3 stars":
            sentiment = "Neutral"
        else:
            sentiment = "Positive"

        return {
            "sentiment": sentiment,
            "confidence": round(score, 3)
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to analyze sentiment. Please try again."
        }
