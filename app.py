from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (so frontend can access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

class ReviewInput(BaseModel):
    review: str

@app.post("/analyze")
def analyze_sentiment(input: ReviewInput):
    inputs = tokenizer(input.review, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    stars = torch.argmax(scores).item() + 1
    confidence = round(scores[0][stars - 1].item(), 2)
    return {"stars": stars, "confidence": confidence}
