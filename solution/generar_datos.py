import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Ensure the output directory exists
output_dir = "data/ref/"
os.makedirs(output_dir, exist_ok=True)

# 2. Load the original dataset from data/raw/iris.csv
input_path = "data/raw/iris.csv"
try:
  df = pd.read_csv(input_path)
  print(f"[INFO] Dataset successfully loaded from {input_path}")
except FileNotFoundError as e:
  print(f"[ERROR] Original dataset not found at {input_path}: {e}")
  exit()

# 3. Split the dataset:
# 70% for reference (baseline) and 30% simulating current production data
reference_df, current_df = train_test_split(df, test_size=0.3, random_state=42)

# 4. Save the output files inside data/ref/
reference_file_path = os.path.join(output_dir, "reference_data.csv")
current_file_path = os.path.join(output_dir, "current_data.csv")

reference_df.to_csv(reference_file_path, index=False)
current_df.to_csv(current_file_path, index=False)

print(f"[INFO] Reference data saved to: {reference_file_path}")
print(f"[INFO] Current data saved to: {current_file_path}")