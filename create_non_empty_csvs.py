from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "iwildcam-2019-fgvc6"
TRAIN_CSV_PATH = DATASET_DIR / "train.csv"
TRAIN_OUTPUT_PATH = DATASET_DIR / "train_without_empty.csv"
EMPTY_CLASS_ID = 0


# Load the labeled training CSV file.
train_df = pd.read_csv(TRAIN_CSV_PATH)

print(f"Loaded train rows: {len(train_df):,}")

# Remove empty-class rows from train.csv.
train_without_empty_df = train_df[train_df["category_id"] != EMPTY_CLASS_ID].copy()
print(f"Train rows after removing empty class: {len(train_without_empty_df):,}")

# Save the filtered output next to the original training file.
train_without_empty_df.to_csv(TRAIN_OUTPUT_PATH, index=False)

print(f"Wrote {TRAIN_OUTPUT_PATH}")
