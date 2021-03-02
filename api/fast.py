from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict_fare")
def predict_fare(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):
    params = {
            "key": pickup_datetime,
            "pickup_datetime": pickup_datetime,
            "pickup_longitude": float(pickup_longitude),
            "pickup_latitude": float(pickup_latitude),
            "dropoff_longitude": float(dropoff_longitude),
            "dropoff_latitude": float(dropoff_latitude),
            "passenger_count": int(passenger_count)
             }
    X_pred = pd.DataFrame(data = params, index = [0])
    model = joblib.load('model.joblib')
    prediction = model.predict(X_pred)
    return prediction[0]

