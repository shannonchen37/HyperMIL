# WSI Data

This directory contains TCGA-derived clinical tables and reference split files for the cohorts used by HyperMIL.

Included cohorts:

```text
WSI_data/
|-- TCGA-STAD/
|   |-- clinical.tsv
|   |-- splits_seed42/
|-- TCGA-THCA/
|-- TCGA-CHOL/
`-- TCGA-LIHC/
```

The original data source is The Cancer Genome Atlas (TCGA), accessed through the Genomic Data Commons (GDC). Raw whole-slide images are not included in this repository and should be downloaded separately from TCGA/GDC according to the applicable data-use policy.

Each `clinical.tsv` file is prepared from the corresponding TCGA cohort metadata. The training code uses the following fields:

- `cases.submitter_id`
- `demographic.vital_status`
- `demographic.days_to_death`
- `diagnoses.days_to_last_follow_up`
- `diagnoses.ajcc_pathologic_stage`

The `splits_seed42/` directories provide reference five-fold sample lists. The local `summary.txt` files are intentionally excluded from version control.
