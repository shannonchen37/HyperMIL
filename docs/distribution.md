# Distribution

The 400 inference-only `.pt` files contain 1,162,648,496 bytes (about 1.08 GiB). The largest file is 32,252,510 bytes (about 30.8 MiB). Every file is below GitHub's regular Git 100 MiB file limit, but keeping all weights in Git history would make source clones unnecessarily large.

Code, YAML configurations, patient/patch split files, manifests and compact evaluation summaries are tracked in Git. The [v1.0.0 release](https://github.com/shannonchen37/HyperMIL/releases/tag/v1.0.0) distributes the weights in four cohort archives with SHA256 checksums. Run `python scripts/download_models.py` from the repository root to download and verify all cohorts, or add `--cohort stad` to download only STAD.

GitHub permits up to 1,000 assets per release, each under 2 GiB; its documentation specifies no total release-size or bandwidth limit. These release assets do not consume Git LFS storage or download quotas. For this model set an external cloud drive is optional and can serve as a download mirror. A checkpoint archive is not a substitute for feature inputs.

The shared manifest records 137,014,190,080 bytes of feature arrays (127.60 GiB) and 535,383,040 bytes of paired coordinates (0.50 GiB), before graph files and raw WSIs. These input assets belong in external storage or must be regenerated with the documented frozen extractor; they should not be committed to the Git repository. Model download storage and feature-data storage are separate decisions.

The feature and graph assets are not included in v1.0.0. To reproduce the verified numbers, mount the matching frozen inputs and configure `configs/paths.local.yaml`. Replacing features with a different extractor or changing their patch order is not equivalent input data.

For maintainers, `scripts/package_weights.py` creates cohort archives and checksum files outside the source tree; it does not upload them.

References: [GitHub large file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [GitHub release asset limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
