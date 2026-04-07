"""Shared dataset utilities for the iWildCam notebooks."""

from collections import Counter
import json
import os

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms

from configs.config import CFG


def load_annotations(
    json_path: str,
    images_dir: str,
    remove_empty: bool = True,
    empty_category_id: int = CFG.empty_category_id,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    images_lookup = {image["id"]: image for image in data["images"]}
    annotation_lookup: dict[int, int] = {}

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in annotation_lookup:
            annotation_lookup[image_id] = annotation["category_id"]

    rows = []
    for image_id, image_info in images_lookup.items():
        category_id = annotation_lookup.get(image_id, empty_category_id)
        file_name = image_info["file_name"]
        rows.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "full_path": os.path.join(images_dir, file_name),
                "category_id": category_id,
                "location": image_info.get("location", -1),
            }
        )

    dataframe = pd.DataFrame(rows)

    if remove_empty:
        dataframe = dataframe[
            dataframe["category_id"] != empty_category_id
        ].reset_index(drop=True)

    unique_categories = sorted(dataframe["category_id"].unique())
    category_to_label = {
        category_id: index for index, category_id in enumerate(unique_categories)
    }
    label_to_category = {
        label: category_id for category_id, label in category_to_label.items()
    }
    dataframe["label"] = dataframe["category_id"].map(category_to_label)

    return dataframe, category_to_label, label_to_category


def get_category_names(json_path: str) -> dict[int, str]:
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return {category["id"]: category["name"] for category in data.get("categories", [])}


def split_dataset(
    dataframe: pd.DataFrame,
    val_size: float = CFG.val_split,
    test_size: float = CFG.test_split,
    seed: int = CFG.random_seed,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        stratify=dataframe["label"],
        random_state=seed,
    )

    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_transforms(
    image_size: int,
    is_train: bool = True,
    mean: tuple[float, float, float] = CFG.imagenet_mean,
    std: tuple[float, float, float] = CFG.imagenet_std,
) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class IWildCamDataset(Dataset):
    def __init__(
        self, dataframe: pd.DataFrame, transform: transforms.Compose | None = None
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["full_path"]).convert("RGB")
        label = row["label"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int = CFG.num_workers,
    pin_memory: bool = CFG.pin_memory,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = IWildCamDataset(
        train_df, get_transforms(image_size=image_size, is_train=True)
    )
    eval_transform = get_transforms(image_size=image_size, is_train=False)
    val_dataset = IWildCamDataset(val_df, eval_transform)
    test_dataset = IWildCamDataset(test_df, eval_transform)

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        class_counts = Counter(train_df["label"].tolist())
        sample_weights = [1.0 / class_counts[label] for label in train_df["label"]]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def print_class_distribution(
    dataframe: pd.DataFrame,
    label_to_category: dict[int, int],
    category_names: dict[int, str] | None = None,
    title: str = "",
) -> None:
    print(f"Class distribution {title}".strip())
    counts = dataframe["label"].value_counts().sort_index()

    for label, count in counts.items():
        category_id = label_to_category[label]
        category_name = f"cat_{category_id}"
        if category_names is not None:
            category_name = category_names.get(category_id, category_name)
        percentage = 100 * count / len(dataframe)
        print(f"[{label:>2}] {category_name:<25s} {count:>6d} ({percentage:5.1f}%)")
