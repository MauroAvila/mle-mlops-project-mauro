from pathlib import Path
import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline


def train_model():
  print("[INFO] Starting MLflow training pipeline for NYC Green Taxi...")

  # 1. Configurar MLflow con backend SQLite local
  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  mlflow.set_experiment("nyc-green-taxi-duration-experiment")

  # 2. Cargar datos de referencia (entrenamiento) y actuales (validación / prueba)
  ref_path = Path("data/ref/reference_data.csv")
  curr_path = Path("data/ref/current_data.csv")

  if not ref_path.exists() or not curr_path.exists():
    print("[ERROR] Reference or current data files not found. Please run data processing first.")
    return

  print("[INFO] Loading datasets...")
  train_df = pd.read_csv(ref_path)
  val_df = pd.read_csv(curr_path)

  # Asegurar que las ubicaciones sean tratadas como texto para el DictVectorizer
  for col in ["PULocationID", "DOLocationID"]:
    if col in train_df.columns:
      train_df[col] = train_df[col].astype(str)
      val_df[col] = val_df[col].astype(str)

  # 3. Definir características (features) y variable objetivo (target)
  target = "duration"
  features = ["PULocationID", "DOLocationID", "trip_distance"]

  X_train = train_df[features].to_dict(orient="records")
  y_train = train_df[target].values

  X_val = val_df[features].to_dict(orient="records")
  y_val = val_df[target].values

  # 4. Iniciar ejecución en MLflow
  with mlflow.start_run() as run:
    print(f"[INFO] MLflow Run ID: {run.info.run_id}")

    # Registrar parámetros generales
    mlflow.log_param("model_type", "LinearRegression + DictVectorizer")
    mlflow.log_param("train_samples", len(train_df))
    mlflow.log_param("val_samples", len(val_df))

    # 5. Construir y entrenar el Pipeline de Machine Learning
    # DictVectorizer convierte las variables categóricas (One-Hot) y deja pasar las numéricas
    pipeline = make_pipeline(
        DictVectorizer(sparse=True),
        LinearRegression()
    )

    print("[INFO] Training model pipeline...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluar el modelo en validación
    print("[INFO] Evaluating model performance...")
    y_pred = pipeline.predict(X_val)
    
    # Calcular RMSE (Root Mean Squared Error)
    rmse = root_mean_squared_error(y_val, y_pred)
    print(f"[METRIC] Validation RMSE: {rmse:.4f}")

    # Registrar métrica en MLflow
    mlflow.log_metric("val_rmse", rmse)

    # 7. Registrar el modelo entrenado en MLflow
    mlflow.sklearn.log_model(pipeline, "model")
    print("[SUCCESS] Model successfully trained and logged to MLflow!")


if __name__ == "__main__":
  train_model()