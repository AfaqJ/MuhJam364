from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer
import joblib

app = FastAPI()

# Enabling CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Thiss can be set to the origin of the HTML page, i set it to all.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading the trained (fine-tuned) model, tokenizer and label encoder
model = joblib.load('distilbert_model_low_memory.joblib')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-german-cased')

label_encoder = joblib.load('label_encoder.joblib')


class Item(BaseModel):
    text: str



class PredictionResult(BaseModel):
    predicted_label: str

@app.get("/")
def read_root():
    return {"Hello": "Open the html code to start predictions"}

@app.post("/predict", response_model=PredictionResult)
def predict(item: Item):
    # Tokenize and encode the input text
    inputs = tokenizer(item.text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).numpy()

    # Decoding the predicted label
    decoded_label = label_encoder.inverse_transform(predictions)[0]

    return {"predicted_label": decoded_label}
