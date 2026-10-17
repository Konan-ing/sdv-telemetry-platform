""" Ingénierie des caractéristiques pour la préparation des données. """

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, lead, avg


def create_spark_session():
    """
    Crée une session Spark pour l'ingénierie des caractéristiques.
    
    Returns:
        SparkSession: Une instance de SparkSession.
    """
    spark = SparkSession.builder\
        .appName("SDV_Feature_engineering")\
        .config("spark_driver/memory","4g")\
        .getOrCreate()

    return spark


def engineer_features(df):
    """
    Effectue l'ingénierie des caractéristiques sur le DataFrame Spark.
    
    Args:
        df (DataFrame): Le DataFrame Spark contenant les données de télémétrie.
    
    Returns:
        DataFrame: Un DataFrame Spark avec les nouvelles caractéristiques.
    """
    # Création d'une fenêtre temporelle ordonnée par la date
    # dans un vrai projet multi-vehicules, on ajouterait .partitionBy("vehicle_id")

    window_spec = Window.orderBy("timestamp")

    # Fenetre glissante pour calculer la moyenne sur les 5 dernières observations
    window_rolling = Window.orderBy("timestamp").rowsBetween(-5,0)

    print("Création des varaibles temporelles (Features)...")

    df_features = df\
        .withColumn("vibration_delta", col("vibration")-lag("vibration",1).over(window_spec))\
        .withColumn("temp_rolling_avg", avg("engine_temp").over(window_rolling))

    # nettoyage de la premiere ligne qui contiendra forcement un null à cause du lag()
    df_features = df_features.na.drop()

    # on decalle la cible ( Label) dans le temps : le modèle à l'instant T doit deviner s'il y a une anomalie à l'instant T+3  ( simulation de l'avance)

    print("Creation de la cible predictive (Label Shift)...")

    df_features = df_features.withColumn("target_anomaly_future", lead("is_anomaly", 3).over(window_spec))

    # On supprime les dernières lignes qui n'ont pas de futur (target = null)
    df_final = df_features.na.drop()

    return df_final



if __name__ == "__main__":
    spark = create_spark_session()
    processed_data_path = "data/processed/telemetry_data.parquet"
    features_data_path = "data/features_data.parquet"

    # chargement des données nettoyées de la phase 1

    clean_df = spark.read.parquet(processed_data_path)

    # application du feature engineering
    features_df = engineer_features(clean_df)

    # affichage d'un apperçu des données avec les nouvelles caracteristiques
    features_df.select("timestamp", "engine_temp", "vibration", "vibration_delta", "temp_rolling_avg", "target_anomaly_future").show(10)

    # sauvegarde des données

    print(f"Sauvegarde des données avec les nouvelles caractéristiques dans le fichier : {features_data_path}...")
    features_df.write.mode("overwrite").parquet(features_data_path)
    spark.stop()