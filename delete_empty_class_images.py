from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "iwildcam-2019-fgvc6"
TRAIN_CSV_PATH = DATASET_DIR / "train.csv"
TRAIN_IMAGE_DIR = DATASET_DIR / "train_images"
EMPTY_CLASS_ID = 0


parser = ArgumentParser()
parser.add_argument(
    "--apply",
    action="store_true",
    help="Actually delete the empty-class image files.",
)
args = parser.parse_args()


# Read train labels and collect empty-class file names.
train_df = pd.read_csv(TRAIN_CSV_PATH)
empty_image_names = train_df.loc[
    train_df["category_id"] == EMPTY_CLASS_ID,
    "file_name",
].drop_duplicates()

print(f"Empty-class image records: {len(empty_image_names):,}")
print(f"Train image folder: {TRAIN_IMAGE_DIR}")

existing_image_paths = []
missing_image_paths = []

# Match CSV file names to extracted training image files.
for file_name in empty_image_names:
    image_path = TRAIN_IMAGE_DIR / file_name
    if image_path.exists():
        existing_image_paths.append(image_path)
    else:
        missing_image_paths.append(image_path)

print(f"Existing empty-class image files: {len(existing_image_paths):,}")
print(f"Missing empty-class image files: {len(missing_image_paths):,}")

if not args.apply:
    print("Dry run only. Re-run with --apply to delete the files.")
else:
    # Delete only the files listed as empty-class images.
    for image_path in existing_image_paths:
        image_path.unlink()

    print(f"Deleted empty-class image files: {len(existing_image_paths):,}")
