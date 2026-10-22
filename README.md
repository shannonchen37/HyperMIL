# HyperMIL

![HyperMIL pipeline: preprocessing, MGAS structural hypergraphs, PASG semantic hypergraphs, dual-branch HGNNs, adaptive fusion, and patient-level survival prediction](assets/pipeline.png)

## HyperMIL: Adaptive Spatial–Semantic Hypergraph MIL for Survival Prediction from Whole-Slide Histopathology

**Shenrong Chen, Xiangmin Han, Chenggang Yan, Yuhan Gao, and Zongpeng Li**

[**Read the full paper (PDF)**](docs/HyperMIL.pdf) · [Pretrained models](https://github.com/shannonchen37/HyperMIL/releases/tag/v1.0.0) · [Main results](results/table_iii.md) · [Ablation results](results/table_iv.md)

## Overview

Prognostic information in whole-slide histopathology can depend on both spatial tissue organization and similar morphological patterns in distant regions. **HyperMIL** represents these complementary relationships through structural and semantic hypergraphs, allowing a patch to participate in multiple tissue contexts.

The framework has three main components:

- **Multi-Granularity Adaptive Segmentation (MGAS):** recursively partitions a spatial–feature graph and retains both parent and child regions as overlapping structural hyperedges, preserving coarse and fine tissue contexts.
- **Posterior-Guided Semantic Grouping (PASG):** discovers slide-specific Gaussian-mixture prototypes and connects each patch to its two most similar prototypes through temperature-scaled cosine affinities, forming sparse, soft semantic incidence across spatially separated regions.
- **Dual-branch learning and adaptive fusion:** independent HGNNs and gated-attention pooling encode the two views. A learned scalar gate fuses the WSI representations, which are averaged within each patient before linear risk prediction with a Cox–Efron objective.

The pipeline starts from frozen **512-dimensional ResNet-34 patch features** and patch coordinates. Hypergraph construction is slide-local and label-independent. Here, semantic groups describe feature-space organization rather than validated tissue phenotypes.

## Experiments and results

The study includes **1,291 patients and 1,344 diagnostic WSIs** from four TCGA cohorts, with overall survival as the endpoint. All slides belonging to a patient remain in the same fold.

| Cohort | Patients | WSIs | Observed events |
| --- | --- | --- | --- |
| TCGA-STAD | 387 | 413 | 163 |
| TCGA-THCA | 505 | 518 | 16 |
| TCGA-CHOL | 39 | 39 | 19 |
| TCGA-LIHC | 360 | 374 | 128 |

### Main comparison

Patient-level Harrell C-index is reported as **mean ± sample standard deviation** over five folds. Macro is the mean of the four unrounded cohort means. Bold denotes the highest mean in each column.

| Method | STAD | THCA | CHOL | LIHC | Macro |
|---|---|---|---|---|---|
| MeanPool | 0.5630 ± 0.0349 | 0.8726 ± 0.0263 | 0.6597 ± 0.1806 | 0.7164 ± 0.0591 | 0.7029 |
| ABMIL | 0.5759 ± 0.0632 | 0.8563 ± 0.0565 | 0.6555 ± 0.2126 | 0.7146 ± 0.0762 | 0.7006 |
| DSMIL | 0.4965 ± 0.0605 | 0.8296 ± 0.0559 | 0.6557 ± 0.2393 | 0.6932 ± 0.0833 | 0.6688 |
| ACMIL | 0.5703 ± 0.0621 | 0.8341 ± 0.0657 | 0.7087 ± 0.1855 | 0.7024 ± 0.0702 | 0.7039 |
| DTFD-MIL | 0.5639 ± 0.0581 | 0.8522 ± 0.0719 | 0.7275 ± 0.1376 | 0.7028 ± 0.0833 | 0.7116 |
| TransMIL | 0.6086 ± 0.0300 | 0.8848 ± 0.0512 | 0.6667 ± 0.1687 | 0.7131 ± 0.0774 | 0.7183 |
| AttriMIL | 0.5413 ± 0.0377 | 0.8101 ± 0.0639 | 0.6983 ± 0.2791 | 0.7004 ± 0.0701 | 0.6875 |
| DualGCN-MIL | 0.5693 ± 0.0595 | 0.8844 ± 0.0432 | 0.6212 ± 0.1611 | 0.7217 ± 0.0638 | 0.6991 |
| PatchGCN | 0.5867 ± 0.0565 | 0.8974 ± 0.0196 | 0.6992 ± 0.2091 | 0.7216 ± 0.0694 | 0.7262 |
| Spatial-KMeans-GroupPool | 0.5558 ± 0.0366 | 0.8698 ± 0.0278 | 0.6597 ± 0.1806 | 0.7166 ± 0.0612 | 0.7005 |
| Semantic-KMeans-GroupPool | 0.5589 ± 0.0246 | 0.8628 ± 0.0239 | 0.6735 ± 0.2030 | 0.7152 ± 0.0594 | 0.7026 |
| IIHGC | 0.5624 ± 0.0364 | 0.8404 ± 0.0467 | **0.7471 ± 0.0738** | 0.7120 ± 0.0841 | 0.7154 |
| Hyper-AdaC | 0.5663 ± 0.0304 | 0.7946 ± 0.1256 | 0.7315 ± 0.1101 | 0.6214 ± 0.0899 | 0.6784 |
| **HyperMIL (Ours)** | **0.6200 ± 0.0257** | **0.9310 ± 0.0254** | 0.7371 ± 0.1202 | **0.7473 ± 0.0629** | **0.7589** |

HyperMIL obtains the highest observed Macro of **0.7589**, with the highest mean on STAD, THCA, and LIHC and the second-highest mean on CHOL. These are point-estimate rankings; the paper reports paired bootstrap uncertainty and discusses checkpoint-selection and external-validation limitations.

AttriMIL omits its classification-specific rank constraint for survival prediction. DualGCN-MIL is a paper-guided reimplementation.

### Ablation study

The six variants test the structural branch, semantic branch, structural hierarchy, semantic assignments, adaptive fusion, and hypergraph propagation. The full pipeline has higher observed means than every variant on all four cohorts.

| Method | STAD | THCA | CHOL | LIHC | Macro |
|---|---|---|---|---|---|
| A1: Structure-only | 0.5738 ± 0.0526 | 0.8594 ± 0.0425 | 0.6899 ± 0.1835 | 0.7043 ± 0.0832 | 0.7068 |
| A2: Semantic-only | 0.5716 ± 0.0365 | 0.8596 ± 0.0554 | 0.6926 ± 0.2143 | 0.6953 ± 0.0602 | 0.7048 |
| A3: Leaf-only | 0.5751 ± 0.0568 | 0.8694 ± 0.0311 | 0.7056 ± 0.1666 | 0.7055 ± 0.0496 | 0.7139 |
| A4: Hard semantic | 0.5809 ± 0.0532 | 0.8760 ± 0.0314 | 0.6544 ± 0.1236 | 0.7214 ± 0.0572 | 0.7082 |
| A5: Fixed fusion | 0.5806 ± 0.0437 | 0.8711 ± 0.0290 | 0.6910 ± 0.1436 | 0.7138 ± 0.0200 | 0.7141 |
| A6: Incidence-based GroupPool | 0.5912 ± 0.0582 | 0.8778 ± 0.0307 | 0.6872 ± 0.1628 | 0.7303 ± 0.0703 | 0.7216 |
| **A0: Full pipeline** | 0.6200 ± 0.0257 | 0.9310 ± 0.0254 | 0.7371 ± 0.1202 | 0.7473 ± 0.0629 | 0.7589 |

The [full paper](docs/HyperMIL.pdf) includes the method formulation, Kaplan–Meier analyses, paired comparisons, and discussion. The [evaluation protocol](docs/protocol.md) describes the released inference pipelines and historical checkpoint selection.

## Code and pretrained models

This release provides HyperMIL, **13 baselines**, and **six ablation variants** through a shared YAML-configured framework. Fresh inference from all **400 model files** reproduced all **105 cells** in Tables III and IV at the reported four-decimal precision. See the [verification report](docs/release_verification.json).

## Quick start

### 1. Install

Use Python 3.10 and a CUDA-compatible PyTorch environment.

```bash
git clone https://github.com/shannonchen37/HyperMIL.git
cd HyperMIL
python -m pip install -r requirements.txt
```

The evaluated environment used PyTorch 2.6.0 with CUDA 11.8. Exact tested versions are listed in [requirements-tested.txt](requirements-tested.txt); install the PyTorch build appropriate for your CUDA environment.

### 2. Download pretrained models

```bash
python scripts/download_models.py
# Download only one cohort:
python scripts/download_models.py --cohort stad
```

The [v1.0.0 release](https://github.com/shannonchen37/HyperMIL/releases/tag/v1.0.0) provides four cohort archives, totaling approximately **1.08 GiB**. The script verifies SHA256 hashes and installs weights into `checkpoints/`. Available cohort options are `stad`, `thca`, `chol`, and `lihc`.

Each method/cohort/fold has one self-contained `.pt` file, such as `hypermil_stad_fold0.pt` or `ablation_a1_stad_fold0.pt`. Multi-member models retain all member weights and combination settings in that file; IIHGC retains both stages. File hashes and fold metrics are recorded in [checkpoints/manifest.csv](checkpoints/manifest.csv).

### 3. Configure data paths

Evaluation requires the frozen **512-D patch features**, aligned patch coordinates, and method-specific graph/group inputs. These data assets are separate from the model download; weights alone are insufficient to run evaluation.

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
```

Edit `configs/paths.local.yaml` to point to your data directories:

```yaml
wsi_data: /path/to/WSI_data
hypergraphs: /path/to/hypergraphs
patchgcn: /path/to/patchgcn_graphs
knn: /path/to/KNN
kmeans: /path/to/kmeans_groups
```

Features and coordinates are loaded from `WSI_data/TCGA-{cohort}/patch_ft/` and `patch_coor/`, using filenames in [manifests/feature_manifest.csv](manifests/feature_manifest.csv). Preserve their patch order and feature representation. See [data and model distribution](docs/distribution.md) for storage requirements.

### 4. Evaluate

Run HyperMIL on one fold:

```bash
python scripts/evaluate.py --config configs/hypermil.yaml --cohort STAD --fold 0
```

Omit `--fold` to evaluate all five folds of that cohort. Available cohorts are `STAD`, `THCA`, `CHOL`, and `LIHC`; the default device is `cuda:0`.

To reproduce both paper tables across all methods and cohorts:

```bash
python scripts/reproduce.py
# Optionally select GPUs:
python scripts/reproduce.py --gpus 0,1
```

The script runs checkpoint inference and resumes completed runs when their recorded code/configuration hashes match. Patient risks and fold reports are written to `results/eval/`. Table comparisons appear in `results/paper_comparison.csv`, with completion status in `results/verification_summary.json`. A completed full run exits with failure if table cells remain unresolved or differ at the reported precision.

## Methods and configuration

**All methods share the data loader, patient splits, evaluation engine, and metric implementation.** YAML files select the model interface, input settings, and method-specific parameters.

IIHGC and Hyper-AdaC use a fixed 2,000-patch subset per WSI (following their original settings), while HyperMIL and other baselines use complete WSIs. All methods share the same code framework, patient splits, and evaluation protocol.

The public [patient split file](splits/patient_folds.csv) is shared by every method. The [patch index file](splits/patch_indices.csv) fixes the subset identities, with at most 2,000 patches per WSI. Evaluation never generates replacement patient splits.

Switch methods by changing the configuration:

```bash
python scripts/evaluate.py --config configs/abmil.yaml --cohort STAD --fold 0
python scripts/evaluate.py --config configs/iihgc.yaml --cohort STAD --fold 0
python scripts/evaluate.py --config configs/ablation_a1.yaml --cohort STAD --fold 0
```

Example input settings from `configs/iihgc.yaml`:

```yaml
method: IIHGC
interface: hypermil.interfaces:IIHGCInterface
data:
  feature_manifest: manifests/feature_manifest.csv
  patient_folds: splits/patient_folds.csv
  max_patches: 2000
  patch_selection: splits/patch_indices.csv
model:
  factory: hypermil.models.iihgc:InterModel
```

This is an excerpt; use the complete supplied configuration to run a method. Full-WSI methods set `max_patches: null`. See [configs/config_index.yaml](configs/config_index.yaml) for all method-to-configuration mappings.

| Ablation | Variant | Configuration |
| --- | --- | --- |
| A0 | Full pipeline | [hypermil.yaml](configs/hypermil.yaml) |
| A1 | Structure-only | [ablation_a1.yaml](configs/ablation_a1.yaml) |
| A2 | Semantic-only | [ablation_a2.yaml](configs/ablation_a2.yaml) |
| A3 | Leaf-only | [ablation_a3.yaml](configs/ablation_a3.yaml) |
| A4 | Hard semantic | [ablation_a4.yaml](configs/ablation_a4.yaml) |
| A5 | Fixed fusion | [ablation_a5.yaml](configs/ablation_a5.yaml) |
| A6 | Incidence-based GroupPool | [ablation_a6.yaml](configs/ablation_a6.yaml) |

## Training

```bash
python scripts/train.py --config configs/abmil.yaml --cohort STAD --fold 0 --output runs/abmil_stad_fold0
```

New runs use the shared trainer with the complete training-patient risk set, Cox loss with Efron correction, and inner-validation checkpoint selection. Model-specific stages are implemented through interfaces. Full-WSI training can require substantial GPU memory.

The released checkpoints reproduce the archived paper results; the shared trainer supports new experiments. Historical checkpoints had fold-specific training and selection settings, including some outer-fold selection, so retraining is not guaranteed to reproduce the released results and these archived scores are not independent test estimates. HyperMIL's default new training run fits a single canonical member, while released models preserve the table-producing pipeline. Details are documented in the [evaluation protocol](docs/protocol.md).

For exact numerical replication, IIHGC's reference GPU is an RTX 2080 Ti. Its graph construction can produce small floating-point differences on other GPU architectures; the reproduction scheduler uses the reference device when available.

## Repository structure

```text
configs/        Method configurations and local data-path template
hypermil/       Shared data loader, training/evaluation engines, interfaces, models
scripts/        Model download, training, evaluation, and table reproduction
splits/         Public patient folds and fixed patch indices
manifests/      Frozen feature manifest
checkpoints/    Model manifest; downloaded .pt files are installed here
results/        Paper tables, comparison summaries, and evaluation outputs
docs/           Evaluation protocol, distribution notes, and verification report
```


## Citation

If you use this work, please cite the accompanying manuscript:

```bibtex
@misc{chen_hypermil,
  title = {HyperMIL: Adaptive Spatial--Semantic Hypergraph MIL for Survival Prediction from Whole-Slide Histopathology},
  author = {Chen, Shenrong and Han, Xiangmin and Yan, Chenggang and Gao, Yuhan and Li, Zongpeng},
  howpublished = {Manuscript},
  url = {https://github.com/shannonchen37/HyperMIL/blob/main/docs/HyperMIL.pdf}
}
```

## License

The HyperMIL source code is released under the MIT License. TCGA-derived clinical tables and split files are provided for research reproducibility and remain subject to TCGA/GDC data-use policies. Upstream model implementations and dependencies retain their respective licenses.
