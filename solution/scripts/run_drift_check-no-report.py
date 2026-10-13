lée y comprende el siguiente código. corrígelo para generar y almacenar el reporte en html en reports/

from datetime import datetime
from pathlib import Path
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


def check_data_drift():
  print("[INFO] Starting data drift monitoring process for NYC Green Taxi...")

  # 1. Rutas de los datos de referencia y actuales
  ref_path = Path("data/ref/reference_data.csv")
  curr_path = Path("data/ref/current_data.csv")

  if not ref_path.exists() or not curr_path.exists():
    print("[ERROR] Reference or current data files not found. Run data processing first.")
    return

  # 2. Cargar los datasets procesados
  reference_df = pd.read_csv(ref_path)
  current_df = pd.read_csv(curr_path)

  # Asegurar que las columnas de IDs geográficos se traten como texto/categóricas
  for col in ["PULocationID", "DOLocationID"]:
    if col in reference_df.columns:
      reference_df[col] = reference_df[col].astype(str)
      current_df[col] = current_df[col].astype(str)

  print(f"[INFO] Reference data shape: {reference_df.shape}")
  print(f"[INFO] Current data shape: {current_df.shape}")

  # 3. Configurar el reporte de Evidently con DataDriftPreset
  drift_report = Report(metrics=[DataDriftPreset()])

  # 4. Ejecutar la evaluación de drift
  print("[INFO] Evaluating drift between reference and current datasets...")
  drift_report.run(reference_data=reference_df, current_data=current_df)

  # 5. Generar reporte alternativo o mostrar estado en consola
  # Almacenar metadatos del reporte o verificar métricas clave
  reports_dir = Path("reports")
  reports_dir.mkdir(parents=True, exist_ok=True)
  
  print("[SUCCESS] Data drift check completed successfully. Metrics evaluated via Evidently.")


if __name__ == "__main__":
  check_data_drift()