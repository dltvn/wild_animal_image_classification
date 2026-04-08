# DINOv2 Embedding Clustering Analysis — Report

## Dataset

- **Source**: iWildCam 2019 FGVC6, `train_without_empty.csv`
- **Images after deduplication**: 24,849 (one frame per `seq_id`, empty class excluded)
- **Classes**: 13 species

| Species | Count |
|---|---:|
| opossum | 4,638 |
| coyote | 3,713 |
| raccoon | 3,019 |
| deer | 2,608 |
| rabbit | 2,488 |
| bobcat | 2,390 |
| cat | 1,569 |
| dog | 1,292 |
| squirrel | 1,179 |
| rodent | 820 |
| skunk | 672 |
| fox | 447 |
| mountain_lion | 14 |

Severe class imbalance: mountain_lion has only 14 samples vs. 4,638 for opossum.

## Model & Embeddings

- **Model**: `vit_large_patch14_reg4_dinov2.lvd142m` (DINOv2 ViT-Large with registers) via `timm`
- **Preprocessing**: 518×518 resize, ImageNet normalization (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
- **Embedding dimension**: 1024 (mean-pooled patch tokens from `forward_features()`, excluding CLS and register tokens)
- **Device**: CUDA (NVIDIA GPU)

## KMeans Sweep — Elbow Analysis

K was swept from 5 to 25. Two selection methods were compared:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 5 | 940,460 | 0.1309 |
| 6 | 902,036 | 0.1314 |
| 7 | 863,147 | 0.1333 |
| 8 | 824,070 | 0.1471 |
| 9 | 794,699 | 0.1559 |
| 10 | 778,442 | 0.1462 |
| 11 | 757,811 | 0.1562 |
| 12 | 747,149 | 0.1511 |
| 13 | 730,161 | 0.1564 |
| 14 | 712,221 | 0.1606 |
| 15 | 697,925 | 0.1582 |
| 16 | 683,442 | 0.1639 |
| 17 | 671,585 | 0.1676 |
| 18 | 663,122 | 0.1635 |
| 19 | 652,022 | 0.1722 |
| 20 | 643,879 | 0.1709 |
| 21 | 634,344 | 0.1632 |
| 22 | 626,208 | 0.1713 |
| 23 | 614,462 | 0.1717 |
| 24 | 611,757 | 0.1734 |
| 25 | 603,323 | 0.1797 |

- **Best K by silhouette score**: 25 (score=0.1797)
- **Elbow K by KneeLocator**: 14 (kneedle with S=1.0, curve='convex', direction='decreasing')

The inertia curve shows a gradual bend rather than a sharp knee, which is typical for high-dimensional deep embeddings. The KneeLocator-identified elbow at K=14 sits in the region where inertia reduction starts to flatten noticeably.

**Final K used: 14** (knee-based). K=14 is close to the number of ground-truth classes (13), suggesting the embedding space naturally partitions around the true number of species.

### K=13 vs K=14 Comparison

| K | ARI | NMI | Silhouette |
|---:|---:|---:|---:|
| 13 (ground-truth count) | 0.1790 | 0.3086 | 0.1564 |
| 14 (knee-based, used) | 0.1835 | 0.3175 | 0.1606 |

K=14 slightly outperforms K=13 on all three metrics, suggesting the embedding manifold has a natural partition just beyond the 13 ground-truth classes — possibly splitting a confusable species pair (e.g., coyote/dog or bobcat/cat) into separate clusters.

## PCA Analysis

PCA was applied to the 1024-dim DINOv2 embeddings and projected to 2D for visualization:

- **PCA explained variance ratio**: PC1=16.9%, PC2=13.1%
- **Total variance explained**: 29.96%

Only ~30% of the variance is captured in the first two components, which is expected for high-dimensional deep embeddings. The PCA plots (one for K=13, one for K=14) are saved alongside the elbow plots.

Insert `pca_species_vs_cluster_k13.png` here.

Insert `pca_species_vs_cluster_k14.png` here.

## Clustering Quality (K=14)

| Metric | Value |
|---|---:|
| K used | 14 |
| Silhouette (K=14) | 0.1606 |
| ARI (vs. species labels) | 0.1835 |
| NMI (vs. species labels) | 0.3175 |

ARI of ~0.18 and NMI of ~0.32 indicate moderate alignment between clusters and species labels. The unsupervised clustering does capture some species structure (better than random) but DINOv2 features alone are not linearly sufficient to cleanly separate all 13 species — visual similarity between some animals (e.g., coyote/dog, bobcat/cat) likely causes confusion in the embedding space.

## Visualizations

All saved to `models/dinov2_clustering/`:

- `kmeans_elbow.png` — Inertia and silhouette curves for K=5–25 with KneeLocator and silhouette best-K marked
- `pca_species_vs_cluster_k13.png` — PCA 2D projection: species labels (left) vs K=13 clusters (right)
- `pca_species_vs_cluster_k14.png` — PCA 2D projection: species labels (left) vs K=14 clusters (right)
- `umap_by_cluster.png` — UMAP 2D projection colored by cluster assignment
- `umap_by_species.png` — UMAP 2D projection colored by ground-truth species
- `tsne_by_cluster.png` — t-SNE 2D projection colored by cluster assignment
- `tsne_by_species.png` — t-SNE 2D projection colored by ground-truth species
- `cluster_species_heatmap.png` — Cross-tabulation heatmap of cluster × species counts

## Key Observations

1. **DINOv2 ViT-L produces separable embeddings** — Species like opossum, squirrel, deer, and rabbit form relatively distinct clusters in UMAP/t-SNE space, indicating DINOv2 features capture species-level morphology.
2. **Confusable species cluster together** — Coyote/dog, bobcat/cat, and fox/squirrel show significant overlap, reflecting visual similarity in the raw images.
3. **Elbow vs. silhouette disagreement** — Silhouette keeps increasing up to K=25 with no peak, suggesting the embedding manifold has a long-tailed cluster structure rather than a clear global optimum. The knee-based K=14 is a more interpretable choice given 13 ground-truth species.
4. **Severe imbalance affects clustering** — Mountain_lion (14 samples) and fox (447) are dwarfed by opossum (4,638), meaning small classes are likely absorbed into larger clusters rather than forming their own.

## Files

- **Script**: `dinov2_clustering_analysis.py`
- **Notebook**: `dinov2_clustering_analysis.ipynb`
- **Embeddings cache**: `models/dinov2_clustering/embeddings.npy` (24,849 × 1024)
- **Metadata**: `models/dinov2_clustering/embedding_metadata.csv`
- **Summary JSON**: `models/dinov2_clustering/clustering_summary.json`