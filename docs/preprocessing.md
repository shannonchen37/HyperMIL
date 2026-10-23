# Preparing HyperMIL inputs

## Feature extraction

Install `requirements-preprocessing.txt`, which adds TorchVision, OpenSlide, OpenCV, NetworkX, Pillow, and tqdm. The OpenSlide binary dependency is supplied by `openslide-bin`; alternative platform installation instructions are available in the [OpenSlide documentation](https://openslide.org/api/python/#installing).

Place raw WSIs under `data/raw_wsi/TCGA-STAD/` (or the corresponding cohort directory), then run:

```bash
python scripts/extract_features.py --config configs/preprocessing/features.yaml --cohort STAD
```

The recovered extraction implementation uses a dense level-0 grid of 256 × 256 patches with a stride of 256, an Otsu tissue mask, and a minimum foreground fraction of 0.25. RGB patches are resized to 224 pixels and normalized with the ImageNet mean and standard deviation. An ImageNet-pretrained ResNet-34 in evaluation mode produces 512-dimensional vectors by global average pooling. The backbone remains frozen; TorchVision downloads its pretrained weights on first use if they are not cached.

Outputs preserve patch order:

```text
data/WSI_data/TCGA-STAD/
  patch_ft/<slide_id>_fts.npy       # float32 [N, 512]
  patch_coor/<slide_id>_coors.npy   # level-0 patch origins [N, 2]
```

The script reuses existing coordinate files and skips completed feature files. For a new cohort or slide collection, prepare a feature manifest in the same format as `manifests/feature_manifest.csv` before graph construction. The supplied manifest and patient splits identify the four released cohorts.

## MGAS structural incidence

```bash
python scripts/build_hypergraphs.py --config configs/preprocessing/hypergraphs.yaml --cohort STAD --family MGAS
```

`hypermil/preprocessing/mgas.py` implements the following steps:

1. Divide level-0 coordinates by the configured patch size, 256 pixels. Connect distinct patch positions whose Euclidean distance is at most 2.9 patch widths and whose feature cosine similarity is at least 0.75.
2. In each connected component of size `n`, select `floor(sqrt(n))` seeds by descending component degree, breaking ties by patch index. Grow initial regions with multi-source breadth-first traversal. Isolated nodes become singleton hyperedges.
3. Recursively split each region using the normalized-Laplacian Fiedler vector and its median threshold. Accept a split only if both children have at least two nodes and normalized cut is at most 0.8. Root depth is zero; recursion stops at depth two.
4. Retain the parent region and accepted children as binary hyperedges.

Outputs are written to `data/hypergraphs/STAD_MGAS_V2/<slide_id>.npz`, using `rows`, `cols`, `values`, and `shape`. The directory name matches the shared evaluation loader. A JSON sidecar records the hierarchy and build settings.

## PASG semantic incidence

```bash
python scripts/build_hypergraphs.py --config configs/preprocessing/hypergraphs.yaml --cohort STAD --family PASG
```

`hypermil/preprocessing/gmm.py` and `pasg.py` implement:

1. Slide-local prototype discovery with ten full-covariance Gaussian components, K-means initialization (`n_init=1`, seed 42), and mixture-weight pseudocount 0.1. Numerical stabilization and the float64 fallback are retained from the original implementation.
2. EM stops after at most 100 iterations, or earlier when `abs(L_t - L_previous) <= 0.001 * max(1, abs(L_previous))`.
3. Retain prototypes whose effective posterior support is at least 2.0. Compute cosine similarity between each L2-normalized patch feature and the retained normalized prototype means, select the top two, and apply softmax with temperature 0.05. Remove unused prototype columns.

The final incidence weights are cosine affinities, rather than the GMM posterior responsibilities. Responsibilities determine prototype support; prototype means determine the final top-two connections.

Outputs are written to `data/hypergraphs/STAD_PASG_V2_FINAL/<slide_id>.npz`. They contain `incidence_row`, `incidence_col`, `incidence_weight`, `incidence_shape`, prototype vectors and IDs, and GMM metadata. These are directly readable by the shared evaluation loader.

## EM iterations and training epochs

The two settings are configured separately:

| Setting | YAML | Scope |
| --- | --- | --- |
| `pasg.em_max_iter: 100` | `configs/preprocessing/hypergraphs.yaml` | E/M updates when fitting a GMM to one slide's frozen features |
| `training.epochs: 100` | `configs/hypermil.yaml` | Optimization epochs of the patient survival model |

Graph construction does not invoke the survival trainer or consume survival labels. Training and evaluation load the saved graph files; they do not rerun EM in every epoch.

The released PASG caches record actual EM iteration ranges of 4–38 (STAD, 413 slides), 4–27 (THCA, 518 slides), 5–13 (CHOL, 39 slides), and 4–18 (LIHC, 374 slides). None of the 1,344 slides reached the 100-iteration upper bound.

## Running selected builds

Omit `--family` to build both branches. Use `--slide <slide_id>` to process one manifest entry, `--device cpu` to select CPU execution, or `--output-root data/rebuilt_hypergraphs` to keep rebuilt graphs separate. Existing graph files are retained unless `--overwrite` is explicitly supplied. The original MGAS spectral partition can vary between repeated builds, and PASG weights can exhibit floating-point roundoff. The frozen downloaded graphs are used for exact evaluation of the released checkpoints.
