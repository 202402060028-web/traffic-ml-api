from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Traffic Congestion Model API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        values = np.array([
            data["CarCount"],
            data["BikeCount"],
            data["BusCount"],
            data["TruckCount"]
        ]).reshape(1, -1)

        values = scaler.transform(values)
        prediction = model.predict(values)[0]

        return {"prediction": str(prediction)}

    except Exception as e:
        return {"error": str(e)}
