import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import joblib
import os

def train_model():
    data_path = "data/features_data.parquet"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    print("Chargement des données d'entraînement...")
    df = pd.read_parquet(data_path)
    
    # 1. Séparation Chronologique (Train/Test Split)
    # On prend les premiers 80% du temps pour apprendre, les 20% restants pour tester
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    features = [
        "engine_temp", "battery_voltage", "vibration", 
        "vibration_delta", "temp_rolling_avg"
    ]
    target = "target_anomaly_future"
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # 2. Gestion du déséquilibre des classes (Scale Pos Weight)
    # Les pannes sont rares (ex: 5% du dataset). On force le modèle à y prêter très attention.
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    print("Entraînement du modèle XGBoost (Optimisé pour le Rappel)...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        scale_pos_weight=pos_weight,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 3. Évaluation Métier
    print("\nÉvaluation sur les données de test (Le Futur non vu) :")
    predictions = model.predict(X_test)
    
    print("\nMatrice de Confusion :")
    cm = confusion_matrix(y_test, predictions)
    print(f"Vrais Négatifs (Véhicules sains ok) : {cm[0][0]}")
    print(f"Faux Positifs  (Fausses alertes)    : {cm[0][1]}")
    print(f"Faux Négatifs  (Pannes ratées)      : {cm[1][0]}  <-- Notre ennemi public n°1")
    print(f"Vrais Positifs (Pannes anticipées)  : {cm[1][1]}")
    
    recall = recall_score(y_test, predictions)
    print(f"\nScore de Rappel (Recall) : {recall:.2f} (Objectif : > 0.80)")
    
    # 4. Sauvegarde du modèle pour l'industrialisation
    model_path = f"{model_dir}/xgboost_predictive_maintenance.pkl"
    joblib.dump(model, model_path)
    print(f"\nModèle sauvegardé avec succès dans : {model_path}")

if __name__ == "__main__":
    train_model()