# Latency Models

The project compares five compact models of one measured operation. All main models fit the same 24 calibration shapes. Twelve development shapes select hyperparameters and the official model; test and challenge labels are reserved for evaluation. FP16 and BF16 models are fitted separately.

Implementation: [`linear_perf/models.py`](../linear_perf/models.py). The [protocol](PROTOCOL.md) defines the measurement target and data permissions.

## Inputs, outputs, and units

For `X[M, K] @ W[N, K].T`, define:

$$
F = 2MNK, \qquad Q = b(MK + NK + MN),
$$

where F is the conventional floating-point operation count and b is bytes per element. FP16 and BF16 both use b = 2 here. Q counts an idealized input read, weight read, and output write. It is a shape-derived proxy, not a hardware measurement of DRAM traffic. Cache reuse, tiling, and repeated loads can change actual traffic. The [NVIDIA matrix multiplication guide](REFERENCES.md#matrix-multiplication) explains the arithmetic-intensity reasoning behind these features.

Prediction functions return **milliseconds**. Reports usually display microseconds. With P expressed in TFLOP/s and B in GB/s, the code uses `F / (P * 1e9)` and `Q / (B * 1e6)` to obtain milliseconds.

| ID | Prediction structure | Main assumption |
|---|---|---|
| `flop_only` | Fixed term + compute term | One effective arithmetic rate explains shape dependence |
| `roofline_2p` | Maximum of compute and memory terms | Two effective rates, with a fixed zero offset |
| `roofline_3p` | Fixed term + maximum of compute and memory terms | A fixed term plus two effective rates |
| `shape_log_ridge` | Exponentiated regularized linear model of shape features | Shape features explain log latency |
| `roofline_residual_krr` | Two-parameter Roofline times a learned correction | Similar shapes share residual behavior around a physical baseline |

## 1. FLOP-only baseline

In consistent time units:

$$
\hat t = t_0 + F/P.
$$

Both the offset and effective compute rate are fitted. This is not simply a line through the origin. It is a useful baseline for testing whether the byte proxy and richer shape features add predictive information.

The positive parameters are optimized using robust least squares on log prediction errors. A fixed set of starting points reduces sensitivity to an arbitrary initialization. Training fit quality alone is not evidence of held-out accuracy.

## 2. Two-parameter Roofline

$$
\hat t = \max(F/P, Q/B).
$$

The maximum represents a simplified overlap hypothesis: whichever resource term is larger determines the estimate. Fixing the offset to zero reduces the number of fitted parameters. It does not assert that launch overhead is physically zero.

The fit uses positive parameters, three fixed starting points, and a robust log-error objective. If all calibration shapes activate only one branch of the maximum, the other rate is not identifiable from those observations. That condition is reported.

## 3. Three-parameter Roofline

$$
\hat t = t_0 + \max(F/P, Q/B).
$$

The offset can absorb effects the two resource terms do not describe. The fit records sensitivity rank, conditioning, proximity to numerical bounds, and variation among near-equivalent starting-point solutions.

**Fitted P, B, and t0 are effective regression coefficients.** They are not measured peak TFLOP/s, measured memory bandwidth, or a direct measurement of CPU submission overhead. A small fitted error can coexist with weakly constrained parameters. For example, if only the compute branch is active, many bandwidth values can produce nearly identical predictions.

The model also cannot establish a real bottleneck merely by labeling its larger term. Such a hardware claim would require additional measurements under the same workload contract.

## 4. Shape-log Ridge

The target is natural log latency. A linear model is fitted to 15 standardized features:

| Feature group | Features | Count |
|---|---|---:|
| Dimensions | `log2(M)`, `log2(N)`, `log2(K)` | 3 |
| Work/data balance | `log2(F/Q)`, `log2(Q)` | 2 |
| Aspect ratio | `log2(N/K)` | 1 |
| Small-M behavior | `exp(-M/32)`, `exp(-M/128)` | 2 |
| Alignment flags | Whether N and K are not divisible by 8 or 16 | 4 |
| Padding proxies | `1 - D / (128 * ceil(D/128))`, for D = M, N, K | 3 |

The 128-based features are fixed geometric proxies. The experiment does not measure or infer an actual 128-element tile choice from them.

For standardized features z, the model is:

$$
\log \hat t = a + z^T\beta.
$$

Feature means and scales are computed from the calibration set alone and stored with the model. The centered log-target is fitted with an L2 penalty on coefficients; the intercept is not penalized. Development median APE chooses alpha from `0.1, 1, 10, 100`.

Predicting in log space produces positive latency and gives relative differences a natural scale. The implementation clips log predictions to predeclared bounds extending a factor of 100 beyond the observed training-time range. Those bounds prevent extreme numerical output; they are not uncertainty intervals.

## 5. Roofline residual KRR

This model starts with the **two-parameter** Roofline prediction, `t_base`. It learns a correction to the ratio between measured and baseline time:

$$
r_i = \log(t_i / t_{\mathrm{base},i}), \qquad
\hat t(x) = t_{\mathrm{base}}(x)\exp(\hat r(x)).
$$

The residual is multiplicative in time, not an additive number of milliseconds. A prediction of `exp(r) = 1.2`, for example, multiplies the baseline by 1.2.

Five features describe each shape: `log2(M)`, `log2(N)`, `log2(K)`, and separate N/K non-multiple-of-16 flags. Training-only standardization is applied before computing RBF similarity:

$$
k(z,z') = \exp\left(-\frac{\lVert z-z'\rVert^2}{2\ell^2}\right).
$$

Kernel ridge regression solves `(G + alpha * I) w = r`, where G is the pairwise similarity matrix. Here **kernel** means a mathematical similarity function; it is unrelated to an executable CUDA kernel.

The development search covers:

- Alpha: `0.01, 0.1, 1`.
- Length scale: `0.5, 1, 2, 4`.

The correction is clipped to a multiplier between 1/4 and 4. Far from the training features, RBF similarities approach zero, so the residual correction approaches zero and the prediction approaches the Roofline baseline. Neither the fallback nor the multiplier bound guarantees accuracy outside the calibration domain.

## Selection and serialization

The family-specific hyperparameter searches and overall model choice use the same declared rule: development median APE, then development P90 APE, then fixed order. There is no calibration-plus-development refit after selection.

Every model is a JSON-serializable payload containing parameters, feature transforms, calibration identity, and a content digest. A suite records all five models, the complete development scores, hyperparameter trials, and the selected model. Predictions for the held-out shapes are stored before their measurements are collected.

For a newly collected native run, predictions can be computed on a CPU without GPU execution:

```python
import json
from pathlib import Path
from linear_perf.models import predict_model

suite = json.loads(Path("runs/fp16_s01/model_suite.json").read_text())
model = suite["models"][suite["selected_model_id"]]
cases = [{"id": "example", "m": 48, "n": 1536, "k": 1024}]
latency_ms = predict_model(model, cases, suite["dtype"])[0]
print(f"Predicted latency: {latency_ms * 1000:.3f} us")
```

The published snapshot uses its dedicated verifier to preserve the separation between recorded artifact identities and publication packaging. The example above uses the native `model_suite.json` created by a fresh run of this release.

## How to interpret errors

For measured time t and predicted time p:

$$
\mathrm{APE} = 100\frac{|p-t|}{t}.
$$

Median APE describes the middle shape error, not the error of every shape. P90 and worst APE reveal tails. RMSE/MAE expose absolute time error; signed bias uses `prediction - observation`. Pairwise ordering measures whether the model orders shapes correctly, after excluding measured near-ties. A model can rank shapes well and still predict poor absolute latencies.

The release also reports `sqrt(mean(log(p/t)^2))` and `exp(mean(abs(log(p/t))))`. The second is a geometric multiplicative error: 1 is exact; 2 denotes a factor-of-two geometric average error. It is not signed bias.

All metrics are evaluated on distinct aggregated shape targets. Block ranges describe the observed timing spread; they are not calibrated prediction intervals. Model domain flags and bootstrap ranges have similarly limited meanings, described in the [protocol](PROTOCOL.md#8-evaluation-and-uncertainty).
