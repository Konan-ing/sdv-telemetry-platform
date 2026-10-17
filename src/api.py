from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialisation de l'API
app = FastAPI(
    title="API de Maintenance Prédictive SDV",
    description="Micro-service de détection d'anomalies pour la flotte de véhicules",
    version="1.0"
)

# Chargement du modèle entraîné lors de la Phase 3
try:
    model = joblib.load("models/xgboost_predictive_maintenance.pkl")
except Exception as e:
    model = None
    print("Attention: Modèle non trouvé. Avez-vous exécuté la Phase 3 ?")

# Définition du schéma de données attendu (Pydantic valide les entrées)
class TelemetryInput(BaseModel):
    vehicle_id: str
    engine_temp: float
    battery_voltage: float
    vibration: float
    vibration_delta: float
    temp_rolling_avg: float

@app.get("/")
def read_root():
    return {"status": "En ligne", "service": "SDV Predictive Maintenance API"}

@app.post("/predict")
def predict_anomaly(data: TelemetryInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    # Transformation des données d'entrée en DataFrame (format attendu par XGBoost)
    input_data = pd.DataFrame([{
        "engine_temp": data.engine_temp,
        "battery_voltage": data.battery_voltage,
        "vibration": data.vibration,
        "vibration_delta": data.vibration_delta,
        "temp_rolling_avg": data.temp_rolling_avg
    }])
    
    # Prédiction de la probabilité de panne
    probability = float(model.predict_proba(input_data)[0][1])
    
    # Règle métier : Si probabilité > 40%, on lève une alerte critique
    is_critical = bool(probability > 0.40)
    
    return {
        "vehicle_id": data.vehicle_id,
        "failure_probability": round(probability * 100, 2),
        "alert_status": "CRITICAL" if is_critical else "NORMAL",
        "action_required": "Schedule maintenance within 48h" if is_critical else "None"
    }