import os
import time
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Inicializar FastAPI
app = FastAPI(
    title="NYC Green Taxi Duration Prediction API",
    description="API de producción con instrumentación Prometheus y registro de eventos para Evidently",
    version="1.0.0"
)

# Configurar URI de MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# URI del modelo en producción según el contrato
MODEL_URI = os.getenv("MODEL_URI", "models:/green-taxi-ride-duration@production")

# Variable global para almacenar el modelo en memoria
model = None

# ==========================================
# INSTRUMENTACIÓN PROMETHEUS
# ==========================================
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Latency in seconds",
    ["endpoint"]
)
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total successful predictions made"
)


@app.on_event("startup")
def load_model_to_memory():
  global model
  print(f"[INFO] Connecting to MLflow at {MLFLOW_TRACKING_URI}")
  try:
    print(f"[INFO] Loading model from URI: {MODEL_URI}")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("[SUCCESS] Model loaded successfully into memory from production alias.")
  except Exception as e:
    print(f"[WARNING] Could not load from Model Registry alias: {e}")
    # Fallback de seguridad al último run registrado
    try:
      client = mlflow.tracking.MlflowClient()
      experiments = client.search_experiments()
      if experiments:
        runs = client.search_runs(
            experiment_ids=[experiments[0].experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        if runs:
          run_id = runs[0].info.run_id
          fallback_uri = f"runs:/{run_id}/model"
          model = mlflow.pyfunc.load_model(fallback_uri)
          print(f"[SUCCESS] Fallback model loaded from run {run_id}")
    except Exception as inner_e:
      print(f"[ERROR] Critical: Failed to load any model. Details: {inner_e}")


# ==========================================
# ESQUEMAS PYDANTIC
# ==========================================
class TaxiTripInput(BaseModel):
  PULocationID: str = Field(..., description="ID de zona de recogida")
  DOLocationID: str = Field(..., description="ID de zona de destino")
  trip_distance: float = Field(..., gt=0.0, le=100.0, description="Distancia del viaje en millas")


class TaxiTripOutput(BaseModel):
  predicted_duration_minutes: float
  model_version_used: str


# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/")
def health_check():
  REQUEST_COUNT.labels(method="GET", endpoint="/", http_status=200).inc()
  return {
      "status": "healthy",
      "model_loaded": model is not None,
      "model_uri": MODEL_URI
  }


@app.post("/predict", response_model=TaxiTripOutput)
def predict_duration(trip: TaxiTripInput):
  start_time = time.time()
  
  if model is None:
    REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=503).inc()
    raise HTTPException(status_code=503, detail="Model not loaded in memory.")

  try:
    # 1. Preparar datos de entrada
    input_data = pd.DataFrame([{
        "PULocationID": str(trip.PULocationID),
        "DOLocationID": str(trip.DOLocationID),
        "trip_distance": float(trip.trip_distance)
    }])

    # 2. Inferencia
    prediction = model.predict(input_data)
    duration_pred = float(prediction[0])

    # 3. Reenvío / Registro de evento para monitoreo (Evidently sink simulado o síncrono)
    # Aquí es donde se consolida el registro de características + predicción
    event_payload = {
        "PULocationID": trip.PULocationID,
        "DOLocationID": trip.DOLocationID,
        "trip_distance": trip.trip_distance,
        "duration": duration_pred,
        "timestamp": time.time()
    }
    # (Opcional) Puedes volcar esto a un archivo log o cliente HTTP hacia Evidently Service
    print(f"[MONITORING SINK] Event logged: {event_payload}")

    # Actualizar métricas
    PREDICTION_COUNTER.inc()
    elapsed = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint="/predict").observe(elapsed)
    REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=200).inc()

    return {
        "predicted_duration_minutes": round(duration_pred, 2),
        "model_version_used": MODEL_URI
    }

  except Exception as e:
    REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=500).inc()
    raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/metrics")
def metrics():
  """Endpoint requerido por Prometheus para recolectar métricas operativas."""
  return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)