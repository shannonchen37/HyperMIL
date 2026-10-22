# Distribution

The 400 inference-only `.pt` files contain 1,162,648,496 bytes (about 1.08 GiB). The largest file is 32,252,510 bytes (about 30.8 MiB). Every file is below GitHub's regular Git 100 MiB file limit, but keeping all weights in Git history would make source clones unnecessarily large.

Recommended layout: publish code, YAML configurations, patient/patch split files, manifests and compact evaluation summaries in the Git repository; distribute one weight archive per cohort as GitHub Release assets. Every release asset must remain under 2 GiB. For this model set an external cloud drive is optional; it can serve as a mirror for download convenience. Frozen features, graph inputs and raw WSIs require separate storage and instructions. A checkpoint archive is not a substitute for feature inputs.

The shared manifest records 137,014,190,080 bytes of feature arrays (127.60 GiB) and 535,383,040 bytes of paired coordinates (0.50 GiB), before graph files and raw WSIs. These input assets belong in external storage or must be regenerated with the documented frozen extractor; they should not be committed to the Git repository. Model download storage and feature-data storage are separate decisions.

Use `scripts/package_weights.py` to create cohort archives and checksum files outside the source tree. No upload is performed by this script. Public download URLs should be added only after the actual release assets exist.

References: [GitHub large file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [GitHub release asset limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
