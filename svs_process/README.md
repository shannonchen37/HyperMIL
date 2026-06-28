# svs_process

This directory contains the minimal WSI preprocessing pipeline required for training.

## Files

- `preprocess_wsi.py`: main entry for patch sampling and feature extraction
- `preprocess_config.yaml`: preprocessing configuration
- `run_preprocess.sh`: launcher script

## Usage

1. Set `data_root` and `save_dir` in `preprocess_config.yaml`
2. Place raw slides under `data_root`
3. Run:

```bash
bash svs_process/run_preprocess.sh
```

## Outputs

The preprocessing step writes the following files under `save_dir`:

- `patch_ft/*.npy`
- `patch_coor/*.npy`
- `sampled_vis/*.jpg`

These outputs are consumed by the hypergraph construction and training scripts.
