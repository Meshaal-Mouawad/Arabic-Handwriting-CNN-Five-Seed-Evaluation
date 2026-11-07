# Online Resource 3 — Current experimental record

**Manuscript:** Family Balancing and Hierarchical CNNs for Arabic Handwritten Symbols: A Controlled Five-Seed Evaluation  
**Author:** Meshaal Mouawad  
**Affiliation:** Department of Electrical and Computer Engineering, Mississippi State University  
**Contact:** mm4922@msstate.edu  
**Release:** IJDAR-CNN-2026-10-28-R1  
**Date:** 28 October 2026

This is the current six-configuration experiment, not the earlier Zenodo course-project report. The release packages existing experimental evidence with new navigation and a separate saved-prediction verification script. It does not change the trained models or experimental implementation. `SHA256SUMS.csv` identifies the delivered files.

## 1. Start here: verification without a GPU or image downloads

From this extracted directory:

```bash
python -m pip install -r requirements-verification.txt
python verification/verify_manifest.py --root .
python verification/verify_predictions.py --root . --out verification_rerun
```

The second command checks delivered file identities. The third independently constructs 38-by-38 confusion matrices from all 30 `predictions.npz` files, recalculates all 210 metric rows, recomputes the five paired clean contrasts and the equally weighted H3 retention contrast, checks 30 selected-checkpoint hashes and prediction hashes, and compares all 16 frozen code files to the experimental lock. It also checks the 20-epoch histories and the validation-based checkpoint choice. It uses NumPy confusion counts rather than importing the original model metric function. Existing evidence is never overwritten. Its successful run is supplied under `verification_output/`.

## 2. Configuration and seed map

| Paper arm | Record directory prefix | Intervention |
|---|---|---|
| B0 | `B0_flat_ce` | Flat classifier, pooled cross-entropy |
| B1 | `B1_flat_balanced` | Flat classifier, equal-family loss |
| B2 | `B2_hier_ce` | Hierarchical probabilities, pooled loss |
| B3 | `B3_hier_balanced` | Hierarchical probabilities, equal-family loss |
| P | `P_hier_smooth_worst` | Hierarchical classifier, smooth higher-loss-family emphasis |
| C | `C_flat_matched` | Flat classifier with matched parameter count |

Each prefix has five directories under `record/runs/`, ending in `_seed7`, `_seed17`, `_seed29`, `_seed43`, or `_seed61`. Each directory contains `initial.pt`, `best.pt`, `training_record.json`, `test_results.json`, and `predictions.npz`.

| File or directory | Evidence supplied |
|---|---|
| `record/EXPERIMENT_LOCK.json` | Six configurations, five seeds, hyperparameters, all 16 implementation hashes, partition hashes and selection policy |
| `record/runs/*/training_record.json` | Twenty epoch records, validation metrics, selected epoch, training/validation identities, model parameter count and reported PyTorch version |
| `record/runs/*/best.pt` | Selected model state and its config/seed/epoch and partition metadata |
| `record/runs/*/initial.pt` | Initial model state from the recorded run |
| `record/runs/*/predictions.npz` | Ground truth and class/family-head predictions for clean and six corrupted conditions |
| `record/runs/*/test_results.json` | Full confusion matrices, metrics, selected-checkpoint and prediction hashes |
| `record/prepared_canonical/source_and_split_manifest.json` | CSV identities, transforms, removed rows, duplicate grouping, counts and prepared-file hashes |
| `record/prepared_canonical/*_labels.npz` | Extracted label vectors; these contain no image pixels or writer identities |
| `record/upstream_source_receipts.json` | Dataset download URLs and archive identities |
| `record/PRETRAINING_POLARITY_AUDIT.json` | Recorded digit/letter foreground-background decision |
| `record/partition_integrity_audit.json` | Original processed-image separation check |
| `record/logs/` | Original training and evaluation logs |
| `code/` | Extracted frozen implementation, byte-identical to the experimental lock |
| `locked/Reproducibility_Code.zip` | Original code ZIP, preserved byte-for-byte |
| `results/full_210_metric_rows.csv` | All six configurations × five seeds × seven conditions |
| `results/PAIRED_HELDOUT_STATISTICS.json` | Unrounded paired estimates, seedwise differences, descriptive intervals and first-seed bootstrap |
| `prior_audit/` | Earlier independent saved-evidence and sampled-inference audit, preserved separately |
| `verification/`, `verification_output/` | New record-recalculation script and its output; not training code |

## 3. Endpoints and statistical interpretation

Family macro-F1 is the unweighted average of per-class F1 over the ten digit classes or 28 letter classes, calculated from the full 38-class confusion matrix. False predictions across families therefore contribute errors to both the true and predicted classes. Worst-family macro-F1 is the smaller family mean. Paired descriptive intervals use five training seeds, not 13,360 independent training replications.

The five clean contrasts are B1−B0, B2−B0, B3−B1, P−B3 and B2−C. In the historic JSON, `H3_smooth_worst/clean_worst_family_F1_difference` is a **supplementary clean diagnostic**. The actual H3 endpoint is `H3_smooth_worst/retention_differences/H3_prespecified_mean_across_6_corruptions`. For each seed it averages, with equal weight over all six corruptions, the P-minus-B3 difference in corrupt-minus-clean worst-family F1. These are differences, not ratios. Historic machine-readable key names have not been rewritten.

## 4. Preprocessing and partitions

The frozen importer is `code/research_code/prepare_canonical_csv.py`. Source CSVs have no header. Digits are reconstructed row-wise as 28×28 uint8 images; letters as 32×32 uint8 images, followed by transpose. Neither source is inverted. Each image is resized to 64×64 using Pillow bilinear interpolation, then converted to float32 and divided by 255. Digit labels remain 0–9; source letter labels 1–28 map to 10–37.

Processed-image BLAKE2b-128 fingerprints define exact duplicates. All original test rows are retained. Before the development split, the importer excludes 23 source-training rows matching 12 distinct processed test fingerprints; the original source indices appear in the manifest. Within each class, exact-image groups are shuffled with `numpy.random.default_rng(2026)` and assigned toward a 15% validation target without splitting a group. The resulting counts are 62,403 training, 11,014 validation, and 13,360 test examples. The original prepared files store images and labels, not source-row indices or writer IDs. The included deterministic algorithm and source hashes provide the reconstruction route.

The large source CSV/image archives and prepared image arrays are not duplicated here. Their hashes are recorded; obtain source archives from the URLs in `record/upstream_source_receipts.json`. Third-party data retain their original distribution conditions.

## 5. Reconstruct prepared data and replay a checkpoint

Use a **new empty directory**, not the packaged immutable `record/` directory:

```bash
python -m pip install -r requirements-reexecution.txt
python code/run_real_experiment.py --root ./reproduction --phase prepare
```

Inspect the generated orientation preview and compare counts and the source/partition identities with `record/prepared_canonical/source_and_split_manifest.json`. The reconstruction command requires internet access to the cited public archives. Supplied dependencies identify required packages, not a recovered historical environment lock. Training records report `pytorch_version` as `2.11.0+cu130`; versions of all other runtime libraries were not recorded. Numerical or archive-serialization differences under another stack must be investigated rather than ignored.

Checkpoint inference can then be rerun without new training:

```bash
python code/research_code/benchmark.py evaluate \
  --test reproduction/prepared_canonical/untouched_test.npz \
  --checkpoint record/runs/B0_flat_ce_seed7/best.pt \
  --out reproduction/replay_B0_seed7.json \
  --save-predictions reproduction/replay_B0_seed7.npz --cpu
```

Omit `--cpu` to use an available CUDA runtime. The example covers one selected checkpoint; repeat over the six prefixes and five seeds for full inference replay. The two published source distributions are needed for image inference, but **not** for the saved-prediction metric check in Section 1.

## 6. Training, sampler and corruption mechanics

To reproduce the full protocol in a new root:

```bash
python code/run_real_experiment.py --root ./reproduction --phase train
python code/run_real_experiment.py --root ./reproduction --phase evaluate
```

The controller fixes six arms × five seeds, 20 epochs, batch 128, Adam learning rate 0.001, λ=0.5 and τ=0.2. The validation criterion is the maximum worst-family macro-F1; the first epoch wins ties. `FixedMixedFamilyBatchSampler` retains the natural source mix, draws an epoch permutation using seed plus epoch offset, drops the incomplete final batch, and rejects a complete batch schedule if a batch lacks either family. The same label-based rule is used for every arm. It does not oversample letters.

`benchmark.py:get_corruption` uses a new NumPy generator seeded 1234 for each noise condition, adds float32 Gaussian noise at intensity σ=0.05 or 0.10, and clips to [0,1]. Blur uses `scipy.ndimage.gaussian_filter` with sigma `(0,0,σ,σ)` for σ=0.8 or 1.3 pixels and SciPy's default reflected boundary handling and truncate=4. Contrast is `clip(0.5 + α*(x−0.5),0,1)` at α=0.75 or 0.50. The test order is clean, two noise settings, two blur settings, then two contrast settings. Each method receives the same condition and seed-generated input. These operations are fixed digital tests, not calibrated camera or scanner simulations.

The code `analyze_predictions.py` reproduces statistical records from any complete runs directory:

```bash
python code/research_code/analyze_predictions.py \
  --runs record/runs --out reproduction/paired_statistics.json --bootstrap 500
```

## 7. Record and verification scope

The delivered archive contains the selected-model evidence and its original logs. The new verification recalculates saved predictions and checks recorded identities and selection choices; it does not retrain models. Earlier sampled checkpoint replay and partition-overlap checks are identified as earlier audit evidence under `prior_audit/`. All source experiment files and the frozen code ZIP are preserved unchanged. Only release documentation, label-only extracts, figure-generation files and the new verification script were added for this submission package.
