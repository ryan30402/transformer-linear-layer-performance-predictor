# Retrospective selection review

These tables recompute already published A100 measurements. The old test and challenge labels were known during the methodology revision. The numbers describe known data; they are **not a fresh independent validation of the revised procedure**. No GPU measurements were added.

## Two distinct comparisons

1. **Criterion-only replay:** keep the original five fitted candidates, their 24 calibration labels, and their previously median-tuned hyperparameters unchanged. Change only the final selection criterion on the 12 development cases. This comparison does not retune hyperparameters.
2. **Grouped CV and refit:** explicitly map the 24 calibration plus 12 development rows to 36 training rows. Select hyperparameters and a model family using leave-one-(N,K)-family-out CV across six families, then refit every family's chosen configuration on all 36 rows. Evaluate the old test and challenge rows afterwards. Candidate-pool labels are excluded.

The second comparison also changes shared geometry/wave-proxy features and robust physical fitting. Its differences cannot be attributed to one design change. All historical rows use warm buffer reuse, and each shape has weight 1. These weights represent the released equal-shape benchmark, not production traffic.

## Check of the review's selection claim

| Development selection criterion | Sessions selecting residual KRR |
|---|---:|
| median_ape_pct | 1 / 6 |
| mean_ape_pct | 6 / 6 |
| p90_ape_pct | 6 / 6 |
| max_ape_pct | 6 / 6 |

In each BF16 session, the saved three-parameter roofline candidate has 5 of 12 development cases above 50% APE; four of those cases have N or K not divisible by 16. Its development median is approximately 13%, while its mean is approximately 30%. The unchanged-candidate replay supports the criticism that median selection concealed a consequential tail in this data. It does not establish that the selector is the sole cause of the original errors.

## Old test results, with the comparison separated

All errors below are percentages. The original and criterion-only columns retain the 24-label fits. The CV/refit columns use 36 training labels. Conventional median and P90 are unweighted NumPy percentiles.

| Session | Original median | Criterion-only mean-selector median | CV/refit mean | CV/refit median | CV/refit P90 | CV/refit worst |
|---|---:|---:|---:|---:|---:|---:|
| fp16_s01 | 25.66 | 17.01 | 15.42 | 10.66 | 34.00 | 41.60 |
| fp16_s02 | 26.53 | 16.56 | 15.28 | 10.95 | 33.61 | 41.21 |
| fp16_s03 | 17.18 | 17.18 | 14.70 | 10.28 | 32.57 | 34.94 |
| bf16_s01 | 48.28 | 16.75 | 19.25 | 13.62 | 42.07 | 65.00 |
| bf16_s02 | 48.50 | 16.74 | 19.08 | 13.48 | 41.74 | 64.85 |
| bf16_s03 | 48.35 | 16.44 | 19.07 | 13.52 | 42.19 | 64.79 |

## Old challenge results after CV/refit

| Session | Selected family | Mean APE | Median APE | P90 APE | Worst APE |
|---|---|---:|---:|---:|---:|
| fp16_s01 | roofline_residual_krr | 14.66 | 14.79 | 31.73 | 34.20 |
| fp16_s02 | roofline_residual_krr | 14.44 | 14.95 | 30.18 | 33.45 |
| fp16_s03 | roofline_residual_krr | 16.24 | 14.73 | 30.69 | 32.14 |
| bf16_s01 | roofline_residual_krr | 19.98 | 18.52 | 31.91 | 58.45 |
| bf16_s02 | roofline_residual_krr | 20.00 | 18.97 | 32.11 | 58.16 |
| bf16_s03 | roofline_residual_krr | 19.94 | 18.67 | 32.72 | 58.10 |

The revised selector chooses residual KRR in all six sessions. Test mean and median errors decrease in this retrospective comparison, but the BF16 test worst case remains approximately 65%; BF16 challenge worst cases remain approximately 58%. Lower typical error therefore does not establish reliable tail behavior or deployment readiness.

## Interpretation and provenance

The CV metric is a tuning score, not an unbiased winner accuracy estimate. `cv_refit_all_models.csv` retains every candidate, pooled out-of-fold scores, fold mean/sample standard deviation, and full test/challenge tails. Fold spread is descriptive and is not a confidence interval. Repeated sessions use the same shapes, so they are not independent workload samples.

The existing CPU snapshot verifier passed 39,708 checks. Every file under `results/` remained byte-identical before and after replay. The verification checks retained evidence and does not independently attest to GPU execution.

The revised experiment still needs newly collected workload-derived cases and prospective validation under its declared measurement/cache contract. Neither the old public test set nor this retrospective report can supply that evidence.

## Reproduce

Run from the repository root with the project's NumPy/SciPy dependencies:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/review_selection.py --compact --output runs/review_analysis
```

Omit `--compact` to retain each refitted suite and detailed per-case evaluation in the output directory. The default output is ignored by Git; the concise reviewed copy is stored here. The script rejects output paths inside the immutable recorded-results tree.

Files: `criterion_only_selection.csv`, `development_candidate_metrics.csv`, `cv_refit_all_models.csv`, `summary.json`, and `source_verification.json`.
