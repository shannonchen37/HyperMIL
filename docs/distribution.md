# Distribution

The 400 inference-only `.pt` files contain 1,162,648,496 bytes (about 1.08 GiB). The largest file is 32,252,510 bytes (about 30.8 MiB). Every file is below GitHub's regular Git 100 MiB file limit, but keeping all weights in Git history would make source clones unnecessarily large.

Code, YAML configurations, patient/patch split files, manifests and compact evaluation summaries are tracked in Git. The [v1.0.0 release](https://github.com/shannonchen37/HyperMIL/releases/tag/v1.0.0) distributes the weights in four cohort archives with SHA256 checksums. Run `python scripts/download_models.py` from the repository root to download and verify all cohorts, or add `--cohort STAD` to download only STAD.

GitHub permits up to 1,000 assets per release, each under 2 GiB; its documentation specifies no total release-size or bandwidth limit. These release assets do not consume Git LFS storage or download quotas. For this model set an external cloud drive is optional and can serve as a download mirror. A checkpoint archive is not a substitute for feature inputs.

## Evaluation inputs

The data release contains 12,096 `.npy`/`.npz` files for four cohorts, totaling 142,866,662,028 bytes (142.87 GB / 133.05 GiB). It is available from [Hugging Face Datasets](https://huggingface.co/datasets/shannonhill/HyperMIL-data) and archived on [Zenodo](https://doi.org/10.5281/zenodo.23567176). Both sources contain the same manifest-listed data files. The source repository does not contain the data files.

| Input | Size (GiB) | Used by |
| --- | ---: | --- |
| Shared patch features | 127.604 | All methods |
| Aligned coordinates | 0.499 | Shared input protocol and spatial methods |
| Spatial/semantic hypergraphs | 0.814 | HyperMIL and its hypergraph variants |
| Compact PatchGCN connectivity | 0.832 | PatchGCN |
| Spatial/feature KNN graphs | 3.251 | DualGCN-MIL |
| GroupPool inputs | 0.054 | Spatial/Semantic-KMeans-GroupPool |

Run `python scripts/download_data.py` to install all inputs, or append `--cohort STAD` for one cohort. The script reads the completed-release marker selected by `configs/dataset.yaml`, checks that it matches the code's manifest, and downloads from the immutable commit named in that marker. It writes only manifest-listed files below `data/`. An explicit immutable revision can also be set in YAML. Interrupted downloads can be resumed by re-running the same command. Downloading original files directly avoids requiring space for both an archive and its extracted contents. Allow additional disk headroom for model weights, runtime files, and evaluation outputs.

For the Zenodo alternative, download all archive parts for each selected cohort and extract them at the code repository root. There are 16 independent tar archives: 4 for STAD, 6 for THCA, 1 for CHOL, and 5 for LIHC. Each archive contains `data/` paths and can be extracted with `tar -xf <archive>.tar`. The record also supplies `data_manifest.csv`, `archive_manifest.json`, and a README. The archives total approximately 142.89 GB, including tar headers; allow space for the archives and the 142.87 GB of extracted inputs. Downloading and extracting only the selected cohort is supported.

The path keys in `configs/paths.yaml` resolve relative to the code repository root. The same loader and method interfaces use these files for training and evaluation. Patient folds and fixed patch selections remain tracked in GitHub. No raw WSIs or new preprocessing/build-graph source code are included in this data release.

The compact PatchGCN interface reads connectivity from `.npz` and shared features from `WSI_data`; no duplicate node-feature tensors are shipped. This representation was evaluated on all four cohorts and all five folds, preserving every fold C-index and the reported table values. It does not change the model architecture or evaluation metric. Legacy `.pt` graph loading remains selectable through YAML.

For maintainers, `scripts/package_weights.py` creates cohort model archives and checksum files outside the source tree; it does not upload them.

References: [Hugging Face dataset downloads](https://huggingface.co/docs/hub/datasets-downloading), [GitHub large file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [GitHub release asset limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
