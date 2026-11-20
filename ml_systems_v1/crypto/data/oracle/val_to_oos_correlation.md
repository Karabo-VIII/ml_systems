> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments.

# VAL→OOS Correlation Validation (2026-05-23T13:35)

## Test setup
- Use POV-17's val_catalog_engines.parquet
- Validate F36.b finding: is TRAIN→VAL anti-predictive STATISTICALLY ROBUST?

## Spearman correlation

- N = 222 engines
- **Spearman(train_compound, val_compound) = 0.5777**
- p-value = 0.0000
- Positive correlation: TRAIN is mildly predictive (refutes F36.b's negative claim)

## Top-30 overlap
- TRAIN top-30 and VAL top-30 share 16 engines
- Jaccard: 0.364
