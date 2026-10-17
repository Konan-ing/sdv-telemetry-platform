""" ce module sert à faire de l'ingestion de données sous pyspark pour permettre une meilleure manipulation des doonnées et derriere gérer plus efficacement les entrainements de modèles de machine learning"""

# importation des bibliothèques nécessaires

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,when,avg
import h5py

def create_spark_session():
    """
    Crée une session Spark pour l'ingestion de données.
    
    Returns:
        SparkSession: Une instance de SparkSession.
    """
    spark = SparkSession.builder \
        .appName("SDV_Telemetry_ingestion")\
        .config("spark.driver.memory", "4g")\
        .getOrCreate()
    return spark

def load_hdf5_to_spark(spark, file_path):
    """
    Charge un fichier HDF5 dans un DataFrame Spark.
    
    Args:
        spark (SparkSession): La session Spark.
        file_path (str): Le chemin du fichier HDF5.
    
    Returns:
        DataFrame: Un DataFrame Spark contenant les données du fichier HDF5.
    """
    # Lire le fichier HDF5 avec h5py
    with h5py.File(file_path, "r") as f:
        df_pd = pd.DataFrame(
            {
                'timestamp': pd.to_datetime(f['timestamp'][:].astype(str)),
                'engine_temp': f['engine_temp'][:],
                'battery_voltage': f['battery_voltage'][:],
                'vibration' : f['vibration'][:],
                'is_anomaly' : f['is_anomaly'][:]

            })
    return spark.createDataFrame(df_pd)


def clean_telemetry(df):
    """
    Nettoie les données de télémétrie en supprimant les valeurs aberrantes et en gérant les valeurs manquantes.
    
    Args:
        df (DataFrame): Le DataFrame Spark contenant les données de télémétrie.
    
    Returns:
        DataFrame: Un DataFrame Spark nettoyé.
    """
    # suppression des valeurs aberrantes pour la temperature du moteur par exemple des valeurs inferieures à 0 et superieures à 200
    df_cleaned = df.withColumn("engine_temp",when((col("engine_temp")>200)| (col("engine_temp")<0),None).otherwise(col("engine_temp"))).withColumn("vibration",when((col("vibration")>10)| (col("vibration")<0),None).otherwise(col("vibration")))

    # gestion des valeurs manquantes
    """
    Pour cette partie, la première idée qui nous vient à l'esprit est de supprimer les lignes contenant des valeurs manquantes.
    cependant cela pourrait entrainer une perte d'information importante.
    Une autre approche serait de remplacer les valeurs manquantes par la moyenne de la colonne correspondante. cependant, cette approche pourrait introduire un biais dans les données.
    Une autre approche serait d'utiliser l'interpolation pour estimer les valeurs manquantes en fonction des valeurs voisines. Cependant, cette approche pourrait ne pas être appropriée pour toutes les colonnes et pourrait introduire des erreurs si les valeurs voisines ne sont pas représentatives.
    Une autre approche serait d'utiliser des modèles de machine learning pour prédire les valeurs manquantes en fonction des autres colonnes. Cependant, cette approche pourrait être complexe et nécessiterait des données supplémentaires pour entraîner les modèles.
    """

    # pour cette partie nous allons utiliser la moyenne pour remplacer les valeurs manquantes
    mean_temp = df_cleaned.select(avg("engine_temp")).collect()[0][0]
    mean_vib = df_cleaned.select(avg("vibration")).collect()[0][0]

    # remplacemement des valeurs manquantes par la moyennne

    df_imputed = df_cleaned.na.fill({"engine_temp": mean_temp, "vibration": mean_vib})
    
    return df_imputed


if __name__ == "__main__":
    # Creation de la session spark

    spark = create_spark_session()

    # chargement du fichier hdf5 dans un dataframe spark
    raw_data_path = "data/raw/telemetry.h5"
    processed_data_path = "data/processed/telemetry_data.parquet"

    raw_df = load_hdf5_to_spark(spark, raw_data_path)
    # nettoyage des données
    clean_df = clean_telemetry(raw_df)

    # Ecriture en parquet pour optimiser I/O de la phase de Machine Learning

    clean_df.write.mode("overwrite").parquet(processed_data_path)
    spark.stop()