from datetime import datetime
from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def run_evidently_monitoring_batch():
  print("[INFO] Starting Evidently AI monitoring service check...")

  # 1. Cargar datos de referencia (baseline)
  ref_path = Path("data/ref/reference_data.csv")
  curr_path = Path("data/ref/current_data.csv")

  if not ref_path.exists() or not curr_path.exists():
    print("[ERROR] Reference or current datasets not found for monitoring.")
    return

  reference_df = pd.read_csv(ref_path)
  current_df = pd.read_csv(curr_path)

  # Asegurar tipos categóricos para los IDs geográficos
  for col in ["PULocationID", "DOLocationID"]:
    if col in reference_df.columns:
      reference_df[col] = reference_df[col].astype(str)
      current_df[col] = current_df[col].astype(str)

  print(f"[INFO] Monitoring window loaded: Reference rows={len(reference_df)}, Current rows={len(current_df)}")

  # 2. Configurar el reporte de Evidently con DataDriftPreset
  print("[INFO] Evaluating Data Drift using Evidently...")
  monitoring_report = Report(metrics=[DataDriftPreset()])

  # 3. Ejecutar la evaluación de monitoreo
  monitoring_report.run(
      reference_data=reference_df,
      current_data=current_df
  )

  # 4. Guardar el reporte de monitoreo en HTML usando save_html()
  reports_dir = Path("reports")
  reports_dir.mkdir(parents=True, exist_ok=True)
  
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  report_filename = reports_dir / f"monitoring_report_{timestamp}.html"

  # Método correcto para exportar en Evidently
  monitoring_report.save_html(str(report_filename))
  
  print(f"[SUCCESS] Monitoring report generated and saved to: {report_filename}")


if __name__ == "__main__":
  run_evidently_monitoring_batch()