import numpy as np
import pandas as pd
import h5py
import os

# Configuration
NUM_RECORDS = 50000
OUTPUT_FILE = "data/raw/telemetry.h5"

def generate_data(n):
    np.random.seed(42)
    # Simulation d'un flux d'une journée
    timestamps = pd.date_range(start="2026-09-12 00:00:00", periods=n, freq="s")
    
    # Capteurs nominaux
    engine_temp = np.random.normal(90, 5, n)      # Température moteur (°C)
    battery_voltage = np.random.normal(12.5, 0.5, n) # Tension batterie (V)
    vibration = np.random.normal(1.0, 0.1, n)        # Niveau de vibration (G)
    
    # Injection des anomalies matérielles
    anomaly_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
    engine_temp[anomaly_indices] += np.random.uniform(20, 40, len(anomaly_indices)) 
    vibration[anomaly_indices] += np.random.uniform(1.5, 3.0, len(anomaly_indices)) 
    
    # Création du DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "engine_temp": engine_temp,
        "battery_voltage": battery_voltage,
        "vibration": vibration,
        "is_anomaly": 0
    })
    
    # Tag des anomalies pour l'entraînement
    df.loc[anomaly_indices, "is_anomaly"] = 1
    return df

def save_to_hdf5(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Stockage sous forme de datasets HDF5 natifs
        f.create_dataset("engine_temp", data=df["engine_temp"].values)
        f.create_dataset("battery_voltage", data=df["battery_voltage"].values)
        f.create_dataset("vibration", data=df["vibration"].values)
        f.create_dataset("is_anomaly", data=df["is_anomaly"].values)
        f.create_dataset("timestamp", data=df["timestamp"].astype(str).values.astype('S'))
        
    print(f"Dataset HDF5 généré avec succès : {filepath} ({len(df)} événements enregistrés)")

if __name__ == "__main__":
    generate_data(NUM_RECORDS).pipe(save_to_hdf5, OUTPUT_FILE)