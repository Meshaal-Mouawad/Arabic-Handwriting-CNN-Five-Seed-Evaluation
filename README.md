# Family Balancing and Hierarchical CNNs for Arabic Handwritten Symbols

Research companion for *Family Balancing and Hierarchical CNNs for Arabic Handwritten Symbols: A Controlled Five-Seed Evaluation*, submitted to the **International Journal on Document Analysis and Recognition (IJDAR)** on 28 October 2026.

## Study

Joint recognition of 10 Arabic digit classes and 28 isolated Arabic character classes using six configurations (B0, B1, B2, B3, P, C) and five random seeds. The study archives 30 training histories and model runs, with seven evaluation conditions per run (210 evaluation records). Family-balanced and hierarchical methods showed small, inconsistent clean-image changes; controlled noise and contrast transformations produced larger performance losses.

## Data sources

The upstream datasets are **not redistributed** in this repository. Download from their original authors:

- MADBase: https://github.com/mloey/Arabic-Handwritten-Digits-Dataset
- AHCD: https://github.com/mloey/Arabic-Handwritten-Characters-Dataset

Source acquisition, preprocessing, and data-integrity checks are implemented under `code/`. Prepared `.npz` files are regenerable and intentionally excluded.

## Contents

- `code/`: frozen training, dataset preparation, testing, analysis, and orchestration code
- `record/`: experiment lock, progress, logs, 30 run directories, predictions, model states, and data audit metadata
- `results/`: reported metrics and paired statistical comparisons
- `figure_generation/`: scripts and inputs for the results figures
- `verification/`: checks to recalculate and compare saved results
- `requirements-*.txt`: package requirements
- `ESM3_ORIGINAL_README.md`, `RELEASE.json`, `ESM3_ORIGINAL_SHA256SUMS.csv`: original supplementary evidence documentation
- `PUBLIC_RELEASE_SHA256SUMS.csv`: checksum manifest for this extracted public release

## Reproduce

Read `ESM3_ORIGINAL_README.md` for the exact frozen evaluation commands and data layout. Use `requirements-reexecution.txt` for model execution and `requirements-verification.txt` for saved-prediction verification. Download upstream datasets to reconstruct data partitions; preserve the original seed and experiment lock.

## Evidence and provenance

This release is extracted from the journal-submitted `ESM_3.zip` without changing the underlying retained file bytes. The original ESM manifest lists additional redundant archives and earlier audit outputs not duplicated in this repository. The public release manifest covers the retained files. **The journal submission and this repository are distinct artifacts.**

## License and reuse

No software license is granted until the author selects one. The original dataset repositories retain their own licensing terms. Cite the paper and dataset creators when reusing materials.
