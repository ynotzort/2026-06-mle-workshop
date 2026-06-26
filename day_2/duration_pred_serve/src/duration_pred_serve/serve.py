import os
import pickle
from typing import Any

from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from fastapi import FastAPI

# model loading
model_path = os.getenv("MODEL_PATH", "model.pkl")
with open(model_path, "rb") as f_in:
    model: Pipeline = pickle.load(f_in)


# server part
class PredictRequest(BaseModel):
    """Inputs required for the prediction."""

    PULocationID: str = Field(description="PickUp Location ID")
    DOLocationID: str = Field(description="DropOff location ID")
    trip_distance: float = Field(description="Distance in km")


class PredictionResponse(BaseModel):
    """Prediction result."""

    prediction_duration_minutes: float = Field(
        description="Predicted trip duration in minutes"
    )

def prepare_features(predict_request: PredictRequest) -> dict[str, Any]:
    result = predict_request.model_dump()
    result["PULocationID"] = str(result["PULocationID"])
    result["DOLocationID"] = str(result["DOLocationID"])
    result["trip_distance"] = float(result["trip_distance"])

    return result

def predict(model_input: dict[str, Any]) -> float:
    prediction = model.predict(model_input)
    return float(prediction[0])

def post_process_prediction(prediction: float) -> float:
    return prediction

app = FastAPI()


@app.post("/predict")
def predict_endpoint(predict_request: PredictRequest) -> PredictionResponse:
    # preproces the input from the user
    # optionally do feature preprocessing
    model_input = prepare_features(predict_request)
    
    # feed the preprocesed data into the model and get a prediction
    prediction_raw = predict(model_input)

    # post process the prediction
    prediction = post_process_prediction(prediction_raw)

    result = PredictionResponse(
        prediction_duration_minutes=prediction
    )
    # return it to the user
    return result
    


# trip_data = {
#     "PULocationID": "43",
#     "DOLocationID": "238",
#     "trip_distance": 1.16,
# }

# prediction = model.predict(trip_data)[0]
# print(prediction)
