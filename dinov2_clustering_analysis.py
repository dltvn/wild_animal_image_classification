from pathlib import Path
import json
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from PIL import Image
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.manifold import TSNE
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import umap

sns.set_theme(style='whitegrid')


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_SLUG = 'dinov2_clustering'
TIMM_MODEL_NAME = 'vit_large_patch14_reg4_dinov2.lvd142m'
RANDOM_SEED = 42

K_MIN = 5
K_MAX = 25
BATCH_SIZE = 32
NUM_WORKERS = 4

CATEGORY_NAME_MAP = {
    0: 'empty',
    1: 'deer',
    2: 'moose',
    3: 'squirrel',
    4: 'rodent',
    5: 'small_mammal',
    6: 'elk',
    7: 'pronghorn_antelope',
    8: 'rabbit',
    9: 'bighorn_sheep',
    10: 'fox',
    11: 'coyote',
    12: 'black_bear',
    13: 'raccoon',
    14: 'skunk',
    15: 'wolf',
    16: 'bobcat',
    17: 'cat',
    18: 'dog',
    19: 'opossum',
    20: 'bison',
    21: 'mountain_goat',
    22: 'mountain_lion',
}


# ── Setup ─────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

try:
    import google.colab
    running_in_colab = True
except ModuleNotFoundError:
    running_in_colab = False

if running_in_colab:
    data_dir = Path('/content/iwildcam-2019-fgvc6')
    models_root = Path('/content/models')
else:
    data_dir = Path('data/iwildcam-2019-fgvc6')
    models_root = Path('models')

train_csv_path = data_dir / 'train_without_empty.csv'
train_image_dir = data_dir / 'train_images'
embedding_dir = models_root / MODEL_SLUG
embedding_dir.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device: {device}')
print(f'CSV path exists: {train_csv_path.exists()}')
print(f'Image directory exists: {train_image_dir.exists()}')
print(f'Embedding directory: {embedding_dir}')


# ── Load & deduplicate data ───────────────────────────────────────────────────

train_df = pd.read_csv(train_csv_path)
train_df['category_name'] = train_df['category_id'].map(CATEGORY_NAME_MAP)
train_df['image_path'] = train_df['file_name'].map(lambda name: train_image_dir / name)
train_df['image_exists'] = train_df['image_path'].map(Path.exists)

filtered_df = train_df[train_df['image_exists']].copy()
filtered_df = filtered_df[filtered_df['category_id'] != 0].copy()

# Deduplicate by seq_id: keep the first frame of each sequence.
dedup_df = filtered_df.sort_values('frame_num').drop_duplicates(subset='seq_id', keep='first').copy()
dedup_df = dedup_df.reset_index(drop=True)

unique_category_ids = sorted(dedup_df['category_id'].unique())
category_to_index = {category_id: index for index, category_id in enumerate(unique_category_ids)}
class_names = [CATEGORY_NAME_MAP[category_id] for category_id in unique_category_ids]
dedup_df['label_index'] = dedup_df['category_id'].map(category_to_index)

print(f'Images after deduplication: {len(dedup_df):,}')
print(f'Number of classes: {len(class_names)}')
print(dedup_df['category_name'].value_counts())


# ── Preprocessing ─────────────────────────────────────────────────────────────

reference_model = timm.create_model(TIMM_MODEL_NAME, pretrained=True)
data_config = resolve_data_config({}, model=reference_model)
del reference_model

image_size = data_config['input_size'][1:]
resize_size = tuple(image_size)
mean = data_config['mean']
std = data_config['std']

transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

print(f'Image size: {resize_size}')
print(f'Mean: {mean}')
print(f'Std: {std}')


# ── Dataset & DataLoader ──────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        with Image.open(row['image_path']) as image:
            image = image.convert('RGB')
        image = self.transform(image)
        return image, row['file_name'], int(row['label_index'])


dataset = EmbeddingDataset(dedup_df, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == 'cuda',
)


# ── Embedding extraction ──────────────────────────────────────────────────────

embedding_path = embedding_dir / 'embeddings.npy'
metadata_path = embedding_dir / 'embedding_metadata.csv'

if embedding_path.exists() and metadata_path.exists():
    print('Loading cached embeddings...')
    embeddings = np.load(embedding_path)
    metadata_df = pd.read_csv(metadata_path)
    print(f'Loaded embeddings shape: {embeddings.shape}')
else:
    print('Extracting DINOv2 ViT-L embeddings...')
    model = timm.create_model(TIMM_MODEL_NAME, pretrained=True)
    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_filenames = []
    all_labels = []

    with torch.inference_mode():
        for images, filenames, labels in tqdm(dataloader, desc='Extracting embeddings'):
            images = images.to(device, non_blocking=True)
            features = model.forward_features(images)
            # features shape: [B, 1 + n_patches + n_registers, 1024]
            # CLS is index 0; patch tokens are 1 : 1 + n_patches; registers follow.
            n_patches = (resize_size[0] // 14) ** 2  # 1369 for 518x518
            patch_tokens = features[:, 1:1 + n_patches, :]
            pooled = patch_tokens.mean(dim=1)
            all_embeddings.append(pooled.cpu().numpy())
            all_filenames.extend(filenames)
            all_labels.extend(labels.numpy().tolist())

    embeddings = np.concatenate(all_embeddings, axis=0)
    metadata_df = pd.DataFrame({
        'file_name': all_filenames,
        'label_index': all_labels,
    })
    metadata_df['category_name'] = metadata_df['label_index'].map(
        {v: class_names[k] for k, v in enumerate(sorted(metadata_df['label_index'].unique()))}
    )

    np.save(embedding_path, embeddings)
    metadata_df.to_csv(metadata_path, index=False)
    print(f'Saved embeddings: {embeddings.shape}')
    print(f'Saved metadata: {len(metadata_df)} rows')


# ── KMeans sweep (elbow method) ───────────────────────────────────────────────

print('Running KMeans sweep...')
inertias = []
silhouettes = []
k_values = list(range(K_MIN, K_MAX + 1))

for k in tqdm(k_values, desc='KMeans sweep'):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    labels = kmeans.fit_predict(embeddings)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(embeddings, labels)
    inertias.append(inertia)
    silhouettes.append(silhouette)

elbow_df = pd.DataFrame({
    'k': k_values,
    'inertia': inertias,
    'silhouette': silhouettes,
})
elbow_path = embedding_dir / 'kmeans_elbow.csv'
elbow_df.to_csv(elbow_path, index=False)

# Plot elbow curves.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_values, inertias, marker='o', color='steelblue')
axes[0].set_title('Elbow Method: Inertia')
axes[0].set_xlabel('Number of clusters (K)')
axes[0].set_ylabel('Inertia')

axes[1].plot(k_values, silhouettes, marker='o', color='seagreen')
axes[1].set_title('Silhouette Score vs K')
axes[1].set_xlabel('Number of clusters (K)')
axes[1].set_ylabel('Silhouette Score')

for axis in axes:
    axis.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(embedding_dir / 'kmeans_elbow.png', dpi=150)
plt.show()

print(elbow_df)


# ── Final KMeans at best K (highest silhouette) ───────────────────────────────

best_idx_silhouette = int(np.argmax(silhouettes))
best_k_silhouette = k_values[best_idx_silhouette]

# Knee point from inertia using KneeLocator
kneedle = KneeLocator(k_values, inertias, S=1.0, curve='convex', direction='decreasing')
best_k_knee = kneedle.elbow

print(f'Best K by silhouette: {best_k_silhouette} (score={silhouettes[best_idx_silhouette]:.4f})')
print(f'Elbow K by KneeLocator: {best_k_knee}')

# Use the knee point K for final clustering if it's in the tested range,
# otherwise fall back to silhouette-based K.
if best_k_knee is not None and K_MIN <= best_k_knee <= K_MAX:
    best_k = int(best_k_knee)
else:
    best_k = best_k_silhouette
    print(f'Knee point outside range, using silhouette-based K={best_k}')

print(f'Using K={best_k} for final clustering')

final_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_SEED)
cluster_labels = final_kmeans.fit_predict(embeddings)
metadata_df['cluster'] = cluster_labels

ari = adjusted_rand_score(metadata_df['label_index'], cluster_labels)
nmi = normalized_mutual_info_score(metadata_df['label_index'], cluster_labels)
print(f'Adjusted Rand Index (ARI): {ari:.4f}')
print(f'Normalized Mutual Information (NMI): {nmi:.4f}')


# ── UMAP visualization ────────────────────────────────────────────────────────

print('Running UMAP...')
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_SEED)
umap_embeddings = umap_reducer.fit_transform(embeddings)
metadata_df['umap_x'] = umap_embeddings[:, 0]
metadata_df['umap_y'] = umap_embeddings[:, 1]

np.save(embedding_dir / 'umap_embeddings.npy', umap_embeddings)

# UMAP colored by cluster.
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    metadata_df['umap_x'],
    metadata_df['umap_y'],
    c=metadata_df['cluster'],
    cmap='tab20',
    s=3,
    alpha=0.6,
)
plt.title(f'UMAP — colored by cluster (K={best_k})')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig(embedding_dir / 'umap_by_cluster.png', dpi=150)
plt.show()

# UMAP colored by species.
plt.figure(figsize=(12, 10))
species_order = sorted(metadata_df['category_name'].unique())
palette = sns.color_palette('tab20', n_colors=len(species_order))
species_color = {s: palette[i] for i, s in enumerate(species_order)}

for species in species_order:
    mask = metadata_df['category_name'] == species
    plt.scatter(
        metadata_df.loc[mask, 'umap_x'],
        metadata_df.loc[mask, 'umap_y'],
        c=[species_color[species]],
        label=species,
        s=3,
        alpha=0.6,
    )

plt.title('UMAP — colored by species label')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(embedding_dir / 'umap_by_species.png', dpi=150)
plt.show()


# ── t-SNE visualization ──────────────────────────────────────────────────────

print('Running t-SNE (this may take a few minutes)...')
tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, n_jobs=-1)
tsne_embeddings = tsne.fit_transform(embeddings)
metadata_df['tsne_x'] = tsne_embeddings[:, 0]
metadata_df['tsne_y'] = tsne_embeddings[:, 1]

np.save(embedding_dir / 'tsne_embeddings.npy', tsne_embeddings)

# t-SNE colored by cluster.
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    metadata_df['tsne_x'],
    metadata_df['tsne_y'],
    c=metadata_df['cluster'],
    cmap='tab20',
    s=3,
    alpha=0.6,
)
plt.title(f't-SNE — colored by cluster (K={best_k})')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig(embedding_dir / 'tsne_by_cluster.png', dpi=150)
plt.show()

# t-SNE colored by species.
plt.figure(figsize=(12, 10))
for species in species_order:
    mask = metadata_df['category_name'] == species
    plt.scatter(
        metadata_df.loc[mask, 'tsne_x'],
        metadata_df.loc[mask, 'tsne_y'],
        c=[species_color[species]],
        label=species,
        s=3,
        alpha=0.6,
    )

plt.title('t-SNE — colored by species label')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(embedding_dir / 'tsne_by_species.png', dpi=150)
plt.show()


# ── Cluster composition table ─────────────────────────────────────────────────

composition = pd.crosstab(metadata_df['category_name'], metadata_df['cluster'])
print(composition)

plt.figure(figsize=(16, 6))
sns.heatmap(composition, cmap='Blues', annot=True, fmt='d', cbar_kws={'label': 'Count'})
plt.title('Cluster vs Species Composition')
plt.xlabel('Cluster')
plt.ylabel('Species')
plt.tight_layout()
plt.savefig(embedding_dir / 'cluster_species_heatmap.png', dpi=150)
plt.show()


# ── Summary ──────────────────────────────────────────────────────────────────

summary = {
    'model': TIMM_MODEL_NAME,
    'embedding_dim': embeddings.shape[1],
    'n_images_after_dedup': len(metadata_df),
    'k_range': f'{K_MIN}-{K_MAX}',
    'best_k_silhouette': int(best_k_silhouette),
    'best_k_knee': int(best_k_knee) if best_k_knee is not None else None,
    'best_k_used': int(best_k),
    'best_silhouette': float(silhouettes[best_idx_silhouette]),
    'ari': float(ari),
    'nmi': float(nmi),
}
print('Summary:')
print(json.dumps(summary, indent=2, default=str))
summary_path = embedding_dir / 'clustering_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f'\nAll outputs saved to: {embedding_dir}')
