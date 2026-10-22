# Release verification

1. Receive final main and ablation tables, metric definitions, cohort names and rounding conventions.
2. Locate candidate checkpoints using existing experiment records, not target-driven tuning. Confirm method, configuration, fold and selection provenance.
3. Audit patient disjointness, exact split membership, shared feature identities, risk orientation, censoring convention and metric aggregation.
4. Implement one shared train/eval engine with method adapters; record any incompatibility with historical training protocols.
5. Copy selected checkpoints without modifying tensor contents. Use neutral method/cohort/fold names; preserve SHA256 and private source-path mapping outside the public package.
6. Evaluate every fold using the shared engine and publish measured values and comparison with paper values. Any mismatch remains unresolved until explained; do not alter evaluation to force agreement.
7. Publish only verified assets and actual runnable reproduction commands.
