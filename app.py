import joblib
from fastapi import FastAPI
import numpy as np

app = FastAPI()

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Traffic Congestion Model API Running"}

@app.post("/predict")
def predict(data: dict):
    values = np.array(list(data.values())).reshape(1, -1)
    values = scaler.transform(values)
    prediction = model.predict(values)[0]
    return {"prediction": int(prediction)}
