# HyperMIL

Patient-level survival prediction with a shared code framework and configurable model interfaces.

IIHGC and Hyper-AdaC use a fixed 2,000-patch subset per WSI (following their original settings), while HyperMIL and other baselines use complete WSIs. All methods share the same code framework, patient splits, and evaluation protocol.

## Setup

Use Python 3.10 and install `requirements.txt` in your CUDA-compatible environment. Copy `configs/paths.example.yaml` to `configs/paths.local.yaml`, then set the locations of the frozen 512-D features, paired coordinates and graph/group inputs. Model weights and feature assets are distributed separately from Git source history. See `docs/distribution.md`.

The public files `splits/patient_folds.csv` and `splits/patch_indices.csv` define the patient folds and fixed patch identities. Every method uses these files through `hypermil/data.py`. No evaluation command generates a new split. A patient with several WSIs remains one survival observation. Feature and coordinate rows are indexed together.

## Download the models

The [v1.0.0 release](https://github.com/shannonchen37/HyperMIL/releases/tag/v1.0.0) provides all 400 model containers in four cohort archives, about 1.08 GiB in total. Download and verify them with:

```bash
python scripts/download_models.py
# Or download one cohort:
python scripts/download_models.py --cohort stad
```

The script verifies each archive's SHA256 before installing files into `checkpoints/`. Frozen patch features, paired coordinates and graph/group inputs remain separate data dependencies and must be mounted through `configs/paths.local.yaml` before evaluation.

## Verified paper results

Fresh checkpoint inference completed for all 400 method/cohort/fold combinations. All 105 reported cells in Tables III and IV match at the paper's four-decimal display precision. See [Table III](results/table_iii.md), [Table IV](results/table_iv.md), and the [verification report](docs/release_verification.json). The checked-in source uses the same inference modules and model interfaces used for this verification; publication adds download instructions and tooling only.

## Evaluate a released fold

```bash
python scripts/evaluate.py --config configs/hypermil.yaml --cohort STAD --fold 0
python scripts/evaluate.py --config configs/iihgc.yaml --cohort STAD --fold 0
python scripts/evaluate.py --config configs/ablation_a1.yaml --cohort STAD --fold 0
```

Change only the YAML configuration to select another model interface. The evaluation launcher, data loader, patient fold selection, prediction output and Harrell C-index implementation are shared. Model interfaces implement architecture-specific risk prediction; they never calculate the reported metric.

Each method/cohort/fold has exactly one `.pt` model file, for example `hypermil_stad_fold0.pt` and `ablation_a1_stad_fold0.pt`. The HyperMIL file embeds all its member parameters and combination settings. The IIHGC file embeds both model stages. Neither opens another checkpoint during model loading. Fixed graph inputs are data assets, separate from learned weights.

## Reproduce both tables

```bash
python scripts/reproduce.py
# Optional independent-fold GPU concurrency:
python scripts/reproduce.py --gpus 0,1,2,3,4
```

This performs fresh model inference; saved reference predictions never replace inference. Reports and patient risks are written to `results/eval/`. `results/paper_comparison.csv` compares four-decimal means and sample standard deviations to Tables III and IV. The macro value is the arithmetic mean of the four unrounded cohort means. A0 reuses HyperMIL's results and does not duplicate its weights. A mismatch produces a failing exit status. Current completion status is recorded in `results/verification_summary.json` when evaluation is aggregated.

The automatic scheduler uses the GPU preference recorded in YAML when that model is available. IIHGC's reference device is an RTX 2080 Ti: rebuilding its threshold-based graphs on other GPU architectures can cause small floating-point differences. The model weights and patient evaluation protocol are unchanged. Every measured difference is retained in the verification report; a per-fold floating-point difference alone does not fail table-level verification.

## Configuration

Each method has its own YAML under `configs/`. For example:

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

The complete checked-in YAML also supplies the feature dimensions, paths file, checkpoint pattern, model arguments, evaluation settings, training settings and interface parameters. Other methods use `max_patches: null` for complete WSIs. Method factories and interfaces are loaded from the YAML import paths; the training/evaluation launchers contain no method-name dispatch.

## New training runs

```bash
python scripts/train.py --config configs/abmil.yaml --cohort STAD --fold 0 --output runs/abmil_stad_fold0
```

All new runs use `hypermil/training.py`: the complete patient training risk set, Cox-Efron objective and inner-validation checkpoint selection. Model stages are supplied by interfaces. The implementation retains full autograd graphs, so full-WSI training can require substantial memory; eval uses inference mode. This new common trainer does not assert that training from scratch reproduces historically selected checkpoints or their selection settings. HyperMIL's default new training run fits a single canonical member; the released eval models preserve the table-producing multi-member pipeline. See `docs/protocol.md` for the exact evaluation behavior.

Public `checkpoints/manifest.csv` includes released file hashes, sizes and evaluated fold metrics. Original research code, models and results are not modified.

## License

The HyperMIL source code is released under the MIT License. TCGA-derived clinical tables and split files are provided for research reproducibility and remain subject to TCGA/GDC data-use policies. Upstream model implementations and dependencies retain their respective licenses.
