from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


def register_latest_model():
  print("[INFO] Starting model registration process...")

  # 1. Configurar URI de seguimiento de MLflow
  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  client = MlflowClient()

  # 2. Buscar el experimento y el run más reciente
  experiment_name = "nyc-green-taxi-duration-experiment"
  experiment = client.get_experiment_by_name(experiment_name)

  if not experiment:
    print(f"[ERROR] Experiment '{experiment_name}' not found. Run training first.")
    return

  runs = client.search_runs(
      experiment_ids=[experiment.experiment_id],
      order_by=["start_time DESC"],
      max_results=1
  )

  if not runs:
    print("[ERROR] No runs found in the experiment.")
    return

  latest_run_id = runs[0].info.run_id
  print(f"[INFO] Found latest Run ID: {latest_run_id}")

  # 3. Registrar el modelo en el Model Registry
  model_name = "green-taxi-ride-duration"
  model_uri = f"runs:/{latest_run_id}/model"

  print(f"[INFO] Registering model as '{model_name}' from URI: {model_uri}")
  model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

  print(f"[SUCCESS] Model registered successfully! Version: {model_version.version}")

  # 4. Asignar el alias 'production' a esta versión registrada
  client.set_registered_model_alias(
      name=model_name,
      alias="production",
      version=model_version.version
  )
  print(f"[SUCCESS] Alias 'production' assigned to version {model_version.version} of '{model_name}'.")


if __name__ == "__main__":
  register_latest_model()