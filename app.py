from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load model once
sentiment_pipeline = pipeline("sentiment-analysis")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "BERT Sentiment API is running!"}

@app.post("/analyze")
def analyze(request: TextRequest):
    try:
        text = request.text
        if not text:
            return {"error": "No text provided"}
        result = sentiment_pipeline(text)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
