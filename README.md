# HyperMIL

**Adaptive spatial–semantic hypergraph multiple instance learning for survival prediction from whole-slide histopathology.**

[Pretrained models](https://github.com/shannonchen37/HyperMIL/releases/tag/v1.0.0) · [Main results](results/table_iii.md) · [Ablation results](results/table_iv.md) · [Evaluation protocol](docs/protocol.md)

HyperMIL models complementary spatial and semantic relationships among histology patches using two hypergraph branches. The branches combine multi-granularity structural regions and semantic patch groups, followed by attention pooling and adaptive fusion to predict patient-level survival risk.

This repository provides HyperMIL, 13 baselines, and six ablation variants in a shared training and evaluation framework. Methods are selected through YAML configurations. Public patient-level five-fold splits and pretrained models support reproduction on TCGA-STAD, TCGA-THCA, TCGA-CHOL, and TCGA-LIHC.

## Results

Patient-level Harrell C-index, reported as **mean ± sample standard deviation** over five folds. Macro is the mean of the four unrounded cohort means.

| Method | STAD | THCA | CHOL | LIHC | Macro |
| --- | --- | --- | --- | --- | --- |
| HyperMIL | 0.6200 ± 0.0257 | 0.9310 ± 0.0254 | 0.7371 ± 0.1202 | 0.7473 ± 0.0629 | **0.7589** |

See the complete [method comparison (Table III)](results/table_iii.md) and [ablation study (Table IV)](results/table_iv.md). Fresh inference from all **400 released model files** reproduced all **105 table cells** at the reported four-decimal precision. See the [verification report](docs/release_verification.json).

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

## License

The HyperMIL source code is released under the MIT License. TCGA-derived clinical tables and split files are provided for research reproducibility and remain subject to TCGA/GDC data-use policies. Upstream model implementations and dependencies retain their respective licenses.
