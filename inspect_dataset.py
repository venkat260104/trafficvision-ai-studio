
import os
import pandas as pd

csv_path = os.path.join(
    "dataset",
    "traffic sign recognication",
    "test_labels.csv"
)

print("CSV file exists:", os.path.exists(csv_path))

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
else:
    print("Please check the CSV file location.")

