# Utilisation d'une image Python légère
FROM python:3.10-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Copie des dépendances et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Copie du code source et du modèle entraîné
COPY src/api.py ./src/api.py
COPY models/ ./models/

# Exposition du port de l'API
EXPOSE 8000

# Commande de lancement du serveur web
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]