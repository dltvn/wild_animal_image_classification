# iWildCam 2019 — Wildlife Species Classification
## COMP 263 Term Project

### Team Members & Assignments

| Member | Notebook | Experiment | Model |
|--------|----------|------------|-------|
| Everyone | `00_setup_and_eda.ipynb` | Setup | Data download, EDA, splits |
| Helia | `01_efficientnet_b4.ipynb` | Exp 1: Supervised (CNN) | EfficientNet-B4 fine-tune |
| Saaram & Vinicius | `02_vit_b16.ipynb` | Exp 3: State-of-the-Art | ViT-B/16 fine-tune |
| Randy & Denys | `03_dinov2_classification.ipynb` | Exp 2: Unsupervised→Transfer | DINOv2 ViT-L linear probe + fine-tune |
| Carlos | `04_dinov2_clustering.ipynb` | Exp 2: Unsupervised | DINOv2 embeddings + KMeans clustering |
| Everyone | `05_comparison.ipynb` | Analysis | Compare all results |

### How It Maps to the 3 Required Experiments

1. **Supervised Learning (CNN):** EfficientNet-B4 — a CNN fine-tuned with varying hyperparameters (LR, weight decay, augmentation).
2. **Unsupervised Learning (Feature Extraction + Transfer):** DINOv2 (self-supervised, no labels used for pretraining) — used for both (a) feature extraction → linear probe classification and (b) KMeans clustering on raw embeddings.
3. **State-of-the-Art:** ViT-B/16 (Vision Transformer) and DINOv2 fine-tuning — transformer architectures representing current SOTA.

### Dataset

- **iWildCam 2019** from Kaggle (FGVC6 at CVPR19)
- ~293K camera trap images, ~14 species + empty
- We **remove empty images** and focus on animal classification
- Stratified train/val/test split (70/15/15)

### Modeling Stack

These experiments use PyTorch and torchvision as an intentional project choice.
The repository still keeps other packages for related coursework, but the active
image-classification notebooks and shared utilities in this project follow the
PyTorch workflow.

### Running Order

**Step 1: One person runs Notebook 00**
- Verifies the raw dataset layout
- Creates stratified splits saved as CSVs
- Exports shared metadata used by the experiment notebooks

**Step 2: Notebooks 01-04 run in parallel**
- Each team member runs their assigned notebook
- All notebooks load the same pre-saved splits and metadata
- **Important:** Notebook 03 must run BEFORE Notebook 04 (embeddings needed)

**Step 3: Everyone runs Notebook 05**
- Loads all saved result JSONs
- Produces comparison charts

### Setup Instructions

1. Install dependencies with UV from the repository root.
2. Place the iWildCam data under `data/raw/` locally, or under `content/data/raw/` in Colab.
3. Run `notebooks/00_setup_and_eda.ipynb` to validate the dataset and create the shared splits.
4. Run the assigned experiment notebook after the split artifacts exist.
5. Run `notebooks/05_comparison.ipynb` after the experiment outputs are available.

### Dependencies

Install dependencies through the repository environment instead of inside the notebooks.

Primary experiment dependencies:
- `torch`, `torchvision`, `timm` (models)
- `scikit-learn` (metrics, clustering)
- `umap-learn` (dimensionality reduction)
- `matplotlib`, `seaborn` (visualization)
- `Pillow` (image loading)

### Colab GPU Requirements

| Notebook | Min GPU | Recommended | Est. Time |
|----------|---------|-------------|-----------|
| 00 (EDA) | None | None | 20 min (download) |
| 01 (EfficientNet) | T4 (16GB) | T4 | 2-4 hrs |
| 02 (ViT-B/16) | T4 (16GB) | L4 (24GB) | 3-6 hrs |
| 03 (DINOv2) | T4 (16GB) | A100 (40GB) | 1-2 hrs (embed) + 4-8 hrs (fine-tune) |
| 04 (Clustering) | CPU | CPU | 15-30 min |
| 05 (Compare) | None | None | 1 min |

### Data Paths

- Local runs use the repository-root `data/` directory.
- Colab runs use `content/data/`.
- Shared utilities resolve these locations so notebooks do not need Drive-specific paths.

### Project Structure

```
iwildcam_project/
├── notebooks/
│   ├── 00_setup_and_eda.ipynb
│   ├── 01_efficientnet_b4.ipynb
│   ├── 02_vit_b16.ipynb
│   ├── 03_dinov2_classification.ipynb
│   ├── 04_dinov2_clustering.ipynb
│   └── 05_comparison.ipynb
├── utils/
│   ├── dataset.py        # Shared data loading & transforms
│   └── training.py       # Shared training loop & metrics
├── configs/
│   └── config.py         # Centralized hyperparameters
├── data/
│   ├── raw/
│   │   ├── train_images/
│   │   ├── test_images/
│   │   └── train_annotations.json
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── metadata.json
│   └── ...
├── results/
│   ├── *_report.json
│   ├── *_confusion_matrix.png
│   ├── *_training_curves.png
│   └── model_comparison.png
├── embeddings/
│   └── dinov2_*.npz
├── checkpoints/
│   └── *.pth
└── README.md
```
