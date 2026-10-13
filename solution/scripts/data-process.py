from pathlib import Path
import pandas as pd


def process_green_taxi_data():
  print("[INFO] Starting NYC Green Taxi data transformation pipeline...")
  raw_path = Path("data/raw/green_tripdata_2024-01.parquet")

  if not raw_path.exists():
    print(f"[ERROR] Raw dataset not found at {raw_path}.")
    return

  # 1. Leer el archivo Parquet crudo
  df = pd.read_parquet(raw_path)
  print(f"[INFO] Raw data loaded with shape: {df.shape}")

  # 2. Cálculo de la Variable Objetivo (duration en minutos)
  print("[INFO] Calculating trip duration...")
  df["duration"] = (
      df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
  ).dt.total_seconds() / 60.0

  # 3. Filtrado y Limpieza de Valores Atípicos (Outliers)
  # - Duración entre 1 y 60 minutos
  # - Distancia entre 0.1 y 100 millas
  print("[INFO] Filtering outliers (duration: 1-60 mins, distance: 0.1-100 miles)...")
  df_filtered = df[
      (df["duration"] >= 1) & 
      (df["duration"] <= 60) & 
      (df["trip_distance"] >= 0.1) & 
      (df["trip_distance"] <= 100)
  ].copy()

  # 4. Conversión de Tipos de Datos (Type Casting)
  # Convertir PULocationID y DOLocationID explícitamente a cadena de texto (string)
  print("[INFO] Converting location IDs to string format...")
  df_filtered["PULocationID"] = df_filtered["PULocationID"].astype(str)
  df_filtered["DOLocationID"] = df_filtered["DOLocationID"].astype(str)

  # 5. Selección de Columnas (Contrato de entrenamiento e inferencia)
  # Features: PULocationID, DOLocationID, trip_distance
  # Target: duration
  required_columns = ["PULocationID", "DOLocationID", "trip_distance", "duration"]
  df_final = df_filtered[required_columns].dropna()

  print(f"[INFO] Cleaned dataset shape: {df_final.shape}")

  # 6. División en datos de referencia (baseline) y datos actuales (producción / drift)
  # Dividimos 70% para referencia y 30% para producción
  train_size = int(len(df_final) * 0.7)
  reference_data = df_final.iloc[:train_size]
  current_data = df_final.iloc[train_size:]

  # 7. Almacenar los resultados en data/ref/
  ref_dir = Path("data/ref")
  ref_dir.mkdir(parents=True, exist_ok=True)

  ref_file = ref_dir / "reference_data.csv"
  curr_file = ref_dir / "current_data.csv"

  reference_data.to_csv(ref_file, index=False)
  current_data.to_csv(curr_file, index=False)

  print(f"[SUCCESS] Reference data saved to {ref_file} ({len(reference_data)} rows)")
  print(f"[SUCCESS] Current data saved to {curr_file} ({len(current_data)} rows)")


if __name__ == "__main__":
  process_green_taxi_data()