from pathlib import Path
from urllib.request import urlretrieve


def download_green_taxi_data(year: int = 2024, month: int = 1):
  print("[INFO] Starting data ingestion process for NYC Green Taxi...")

  # 1. Definir la ruta de destino en data/raw/
  color = "green"
  filename = f"{color}_tripdata_{year}-{month:02d}.parquet"
  raw_dir = Path("data/raw")
  data_path = raw_dir / filename

  # 2. Asegurar que el directorio data/raw/ exista
  raw_dir.mkdir(parents=True, exist_ok=True)

  # 3. URL oficial del almacenamiento público de la NYC TLC
  url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"

  # 4. Verificar si ya existe para evitar descargas redundantes
  if data_path.exists():
    print(f"[INFO] Dataset already exists locally at: {data_path}")
    return data_path

  print(f"[INFO] Downloading dataset from {url}...")
  try:
    urlretrieve(url, data_path)
    print(f"[SUCCESS] Dataset successfully downloaded and saved to: {data_path}")
  except Exception as e:
    print(f"[ERROR] Failed to download dataset: {e}")
    raise

  return data_path


if __name__ == "__main__":
  download_green_taxi_data(year=2024, month=1)