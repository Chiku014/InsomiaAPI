import pickle
import numpy as np

# Load the model
with open("app/artifacts/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoders
with open("app/artifacts/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load scaler if used
try:
    with open("app/artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

def preprocess_input(data: dict):
    for col, encoder in label_encoders.items():
        if col in data:
            data[col] = encoder.transform([data[col]])[0]

    # Feature order - must match training
    ordered_features = [
        'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
        'Physical Activity Level', 'Stress Level', 'BMI Category', 'Blood Pressure',
        'Heart Rate', 'Daily Steps'
    ]
    features = [data[col] for col in ordered_features]

    if scaler:
        features = scaler.transform([features])[0]

    return np.array([features])

def predict(data: dict):
    input_data = preprocess_input(data)
    return model.predict(input_data)[0]
