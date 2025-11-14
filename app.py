import joblib
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel


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

class TrafficData(BaseModel):
    time: str
    date: str
    day_of_week: str
    car_count: float
    bike_count: float
    bus_count: float
    truck_count: float
    total: float

