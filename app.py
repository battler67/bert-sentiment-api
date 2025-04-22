from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
import torch

# Create FastAPI instance
app = FastAPI()

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Define input data model
class Review(BaseModel):
    review: str

# Define prediction route
@app.post("/analyze")
async def analyze_sentiment(data: Review):
    result = sentiment_pipeline(data.review)[0]
    label = result["label"]
    score = result["score"]

    # Convert star label to Positive/Negative/Neutral
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

@app.get("/")
async def home():
    return {"message": "BERT Sentiment API is live!"}
