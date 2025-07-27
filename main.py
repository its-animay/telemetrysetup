# demo_log.py

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
from typing import List

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris-logger")

# ---------- ML Setup ----------
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
model = RandomForestClassifier()
model.fit(X_train, y_train)
logger.info("Iris model trained and ready.")

# ---------- FastAPI Setup ----------
app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/ready_check")
def ready_check():
    logger.info("Readiness check passed.")
    return {"status": "ready"}

@app.get("/live_check")
def live_check():
    logger.info("Liveness check passed.")
    return {"status": "alive"}

@app.post("/predict")
def predict(features: IrisFeatures):
    logger.info(f"Prediction request received: {features}")
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    prediction = model.predict(input_data)[0]
    predicted_class = iris.target_names[prediction]
    logger.info(f"Predicted class: {predicted_class}")
    return {"prediction": predicted_class}
