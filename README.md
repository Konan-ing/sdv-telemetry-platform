# SDV Telemetry Platform

**Prototype de maintenance prédictive pour véhicules connectés** — de la télémétrie à l’alerte, avec Python, PySpark, XGBoost, FastAPI et Streamlit.

Ce projet de portfolio illustre la construction d’une chaîne de traitement de données et son exposition sous forme d’API et de tableau de bord. Les données utilisées par le générateur sont **synthétiques** : elles servent à la démonstration et ne représentent pas des mesures réelles de véhicules.

## Ce que fait le projet

1. Génère des événements de télémétrie synthétiques (température moteur, tension batterie, vibrations).
2. Nettoie et prépare les données avec PySpark.
3. Calcule des caractéristiques temporelles, puis entraîne un classifieur XGBoost.
4. Expose une probabilité d’anomalie via une API FastAPI.
5. Présente l’analyse dans une interface Streamlit. Un assistant Hugging Face optionnel est également intégré au parcours de démonstration.

## Architecture

```text
Télémétrie HDF5
      │
      ▼
src/data_ingestion.py ──► données nettoyées (Parquet)
      │
      ▼
src/feature_engineering.py ──► caractéristiques temporelles
      │
      ▼
src/model_training.py ──► modèle XGBoost (.pkl)
      │
      ▼
src/api.py (FastAPI) ◄── app/dashboard.py (Streamlit)
```

## Démarrage rapide

### Prérequis

- Python 3.10 ou supérieur
- Java installé et configuré pour PySpark si vous exécutez le pipeline de données

### Installer les dépendances

Depuis la racine du dépôt :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn requests
```

### Lancer l’API et le tableau de bord

Le modèle de démonstration est fourni dans `models/`. Dans un premier terminal :

```bash
uvicorn src.api:app --reload --port 8000
```

Dans un second terminal, avec le même environnement Python activé :

```bash
streamlit run app/dashboard.py
```

L’interface est disponible à l’adresse affichée par Streamlit. La documentation interactive de l’API est disponible sur <http://127.0.0.1:8000/docs>.

## Exécuter le pipeline de données

Depuis la racine du dépôt, avec Java et les dépendances installés :

```bash
python -m src.generate_telemetry
python -m src.data_ingestion
python -m src.feature_engineering
python -m src.model_training
```

Les fichiers générés sont enregistrés sous `data/` et le modèle entraîné sous `models/`. Les données et sorties de calcul locales ne sont pas suivies par Git.

## Exemple d’appel API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_id": "SDV-001",
    "engine_temp": 95.0,
    "battery_voltage": 12.4,
    "vibration": 1.2,
    "vibration_delta": 0.2,
    "temp_rolling_avg": 90.0
  }'
```

La réponse contient la probabilité estimée de défaillance, un statut (`CRITICAL` ou `NORMAL`) et une action recommandée.

## Configuration de l’assistant

L’assistant Hugging Face est optionnel. Pour le configurer, définissez `HF_API_TOKEN` dans l’environnement avant de lancer Streamlit :

```bash
export HF_API_TOKEN="votre_jeton"
```

Ne placez jamais de jeton ou de secret dans le dépôt. Le fichier local Streamlit de secrets et les fichiers d’environnement sont exclus de Git.

## Structure du dépôt

```text
app/dashboard.py          Interface Streamlit
src/api.py                API de prédiction FastAPI
src/data_ingestion.py     Ingestion HDF5 et nettoyage PySpark
src/feature_engineering.py Caractéristiques temporelles
src/generate_telemetry.py Génération des données synthétiques
src/model_training.py    Entraînement et évaluation XGBoost
models/                   Modèle de démonstration
Dockerfile                Image de l’API
requirements.txt          Dépendances Python
```

## Périmètre

Ce dépôt est un prototype pédagogique et un point de départ pour une application de maintenance prédictive. Les données sont synthétiques et les seuils d’alerte sont des règles de démonstration ; le modèle et les recommandations ne sont pas destinés à piloter une décision de maintenance réelle.
