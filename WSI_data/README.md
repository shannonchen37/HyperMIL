# WSI Data

Dataset files are intentionally not included in this repository.

Create one directory per TCGA cohort, for example:

```text
WSI_data/
|-- TCGA-STAD/
|   |-- clinical.tsv
|   |-- splits_seed42/
|   |-- patch_ft/
|   `-- patch_coor/
|-- TCGA-THCA/
|-- TCGA-CHOL/
`-- TCGA-LIHC/
```

`clinical.tsv` should be prepared locally from the corresponding TCGA cohort metadata. The training code uses the following fields:

- `cases.submitter_id`
- `demographic.vital_status`
- `demographic.days_to_death`
- `diagnoses.days_to_last_follow_up`
- `diagnoses.ajcc_pathologic_stage`

Raw slides, clinical tables, split lists, extracted patch features, and generated hypergraphs should remain local unless your data-use policy explicitly permits redistribution.
