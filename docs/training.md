# Training and checkpoint details

All interfaces read the same frozen feature manifest and patient-level five-fold split. `max_patches` controls input selection in the common loader. IIHGC and Hyper-AdaC select the same fixed indices (at most 2,000 per slide); other interfaces receive every patch. Their method-specific graph or pooling operations remain inside the model interfaces.

Every evaluation runs strict state loading, eval mode and inference mode. A common engine writes one scalar risk per patient and calculates Harrell C using observed earlier events as comparable pairs, with risk ties scoring one half. Five-fold standard deviation uses ddof=1. The reported macro averages the four unrounded cohort means.

The released STAD, THCA and LIHC HyperMIL checkpoints contain multiple member networks, fold-specific signed combination weights and a within-fold percentile transform with average ties in float64. The complete supplied fold defines the percentile reference population. CHOL retains a single network and raw risks. One `.pt` means one self-contained model-weight container; it does not change the number of internal networks or their arithmetic. No source checkpoints or stored patient predictions are needed for this computation.

DTFD uses the fixed validation pseudo-bag seeds specified in YAML. IIHGC uses its saved first-stage parameters and reconstructs second-stage context from training-patient features and first-stage predictions; it does not use evaluation survival outcomes as model inputs. Hyper-AdaC retains its clustering and survival head. Group pooling uses model-independent fixed input aggregation.

The archived table-producing models have historical, fold-specific training and selection settings. Some selected models used the evaluated outer folds for selection. Sharing the new evaluation framework does not make those historical choices identical, or turn these results into independent test estimates. Original settings and source paths are retained in the separate source mapping. New common training uses inner validation. Reproducing archived inference and making a new matched training comparison are separate claims.
