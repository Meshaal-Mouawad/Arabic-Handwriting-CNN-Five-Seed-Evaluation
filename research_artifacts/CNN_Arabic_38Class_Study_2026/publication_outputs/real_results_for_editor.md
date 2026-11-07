# Empirical findings generated from the full real-data experiment

Completed held-out evaluation: 6 model conditions × 5 seeds × 7 test conditions = 210 metric rows.

| Prespecified comparison | Mean clean paired Δ worst-family macro-F1 | 95% descriptive across-seed interval |
|---|---:|---|
| H1_flat_balance | +0.0021 | [-0.0076, +0.0118] |
| H2_hierarchy_CE | +0.0005 | [-0.0137, +0.0148] |
| H2_hierarchy_balanced | -0.0005 | [-0.0126, +0.0115] |
| H3_smooth_worst | -0.0002 | [-0.0142, +0.0137] |
| capacity_H2_vs_matched_flat | -0.0008 | [-0.0097, +0.0082] |

For H3, the pre-specified mean corruption-retention contrast is in the paired-statistics JSON.

The reported intervals quantify between-seed variability; image-bootstrap results are separately reported.
