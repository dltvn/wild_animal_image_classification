"""Shared configuration for the iWildCam experiments."""

from dataclasses import dataclass, field
import os


def _resolve_runtime_root() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    colab_root = os.path.join(os.sep, "content")
    colab_data_dir = os.path.join(colab_root, "data")

    if os.path.isdir(colab_data_dir):
        return colab_root

    return project_root


def _join(*parts: str) -> str:
    return os.path.join(*parts)


@dataclass
class Config:
    runtime_root: str = field(default_factory=_resolve_runtime_root)
    random_seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15
    remove_empty: bool = True
    empty_category_id: int = 0

    img_size_efficientnet: int = 380
    img_size_vit: int = 224
    img_size_dino: int = 224

    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    eff_batch_size: int = 24
    eff_lr: float = 1e-4
    eff_epochs: int = 15
    eff_weight_decay: float = 1e-4

    vit_batch_size: int = 16
    vit_lr: float = 3e-5
    vit_epochs: int = 12
    vit_weight_decay: float = 1e-2

    dino_lp_batch_size: int = 64
    dino_lp_lr: float = 1e-3
    dino_lp_epochs: int = 20

    dino_ft_batch_size: int = 8
    dino_ft_lr: float = 1e-5
    dino_ft_epochs: int = 10
    dino_ft_unfreeze_blocks: int = 4
    dino_ft_weight_decay: float = 1e-2

    cluster_k_values: tuple[int, ...] = (10, 14, 20, 30)
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    tsne_perplexity: int = 30

    num_workers: int = 2
    pin_memory: bool = True
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    early_stopping_patience: int = 5

    def __post_init__(self) -> None:
        self.data_dir = _join(self.runtime_root, "data")
        self.raw_data_dir = _join(self.data_dir, "raw")
        self.processed_data_dir = _join(self.data_dir, "processed")
        self.splits_dir = _join(self.data_dir, "splits")

        self.train_images_dir = _join(self.raw_data_dir, "train_images")
        self.test_images_dir = _join(self.raw_data_dir, "test_images")
        self.train_json = _join(self.raw_data_dir, "train_annotations.json")

        self.results_dir = _join(self.runtime_root, "results")
        self.embeddings_dir = _join(self.runtime_root, "embeddings")
        self.checkpoints_dir = _join(self.runtime_root, "checkpoints")

    def ensure_output_dirs(self) -> None:
        for path in (
            self.processed_data_dir,
            self.splits_dir,
            self.results_dir,
            self.embeddings_dir,
            self.checkpoints_dir,
        ):
            os.makedirs(path, exist_ok=True)


CFG = Config()
