from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import predict

app = FastAPI(title="🛌 Health Oracle API")

class SleepInput(BaseModel):
    Gender: str
    Age: int
    Occupation: str
    Sleep_Duration: float
    Quality_of_Sleep: int
    Physical_Activity_Level: int
    Stress_Level: int
    BMI_Category: str
    Blood_Pressure: str
    Heart_Rate: int
    Daily_Steps: int

@app.post("/predict")
def predict_sleep_disorder(input_data: SleepInput):
    prediction = predict(input_data.dict())
    return {"prediction": prediction}
