# Technical References

The benchmark uses execution and measurement principles documented by PyTorch and NVIDIA. Its exact shape set, repeats, warmup duration, quality thresholds, model menu, and selection rules are project-specific choices defined in the [protocol](PROTOCOL.md).

## Execution and timing

[PyTorch 2.11: CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html) explains asynchronous GPU execution, synchronization/event timing, CUDA Graph capture/replay, static memory addresses, and precision controls. These inform the benchmark's timer boundary and recorded execution policy.

[NVIDIA CUDA C++ Best Practices Guide: Timing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing) provides background on CPU and CUDA event timers and the synchronization needed to interpret them.

## Matrix multiplication

[NVIDIA: Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) describes GEMM operation counts, arithmetic intensity, tiling, and dimension effects. The project uses this reasoning to define FLOP/byte features and interpretable baselines. Its fitted effective rates and alignment proxies are model quantities, not hardware counters.

## Benchmark conditions

[NVIDIA TensorRT: Performance Benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html) discusses timing scope, warmup, reproducible comparisons, and hardware/software conditions including clocks, power, and temperature. The project applies these measurement principles to a PyTorch GEMM experiment; it does not benchmark a TensorRT engine or claim vendor certification.

## Profiling and cache policy

[NVIDIA Nsight Compute: Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) explains profiling overhead, replay, and cache-control effects. This motivates retaining profiler output separately from the primary event-timed labels and documenting cache assumptions explicitly.

## Colab

[Google Colab: Frequently Asked Questions](https://research.google.com/colaboratory/faq.html) documents resource availability and runtime limits. The notebooks inspect and record the assigned device instead of treating a requested accelerator type as a guaranteed experimental environment.

## Project evidence

The external documents explain measurement concepts. Current experiment claims
come from the [completed A100 evidence](../experiments/a100_20261106/README.md),
[selected-model metrics](../experiments/a100_20261106/selected_model_metrics.csv),
[all-model metrics](../experiments/a100_20261106/metrics.csv), and raw quality,
timing, profiling, and model records in its unchanged result archive. The
[historical dataset](../results/README.md) and its
[quality records](../results/summary/quality_and_audits.csv) remain separate.
Official documentation does not establish this project's measured accuracy or
validate its generalization claims.

## Offline model selection and data boundaries

[scikit-learn: Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html) describes fitting transforms and selecting models without test information. This project adopts ordinary offline label isolation; labels do not have to be acquired after fitting.

[scikit-learn: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) describes disjoint groups across training/validation folds. This project implements its own deterministic family grouping and records every fold; it does not depend on scikit-learn.

[scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) documents refitting the selected configuration on all available training data. Here weighted mean APE is a declared project objective, not a universal industry requirement. CV tuning scores remain separate from held-out evaluation.

## Cache-sensitive microbenchmarks

[NVIDIA NVBench](https://github.com/NVIDIA/nvbench/blob/main/README.md) distinguishes measurement contexts including cache effects. The project's four-buffer rotation is an explicit comparison condition; it does not claim NVBench equivalence, guaranteed cold-cache execution, or realistic whole-model traffic.

## Workload architecture sources

Five public Hugging Face model configuration files are pinned by immutable commit and raw SHA256 in [model-source provenance](../configs/model_sources/provenance.json): Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, SmolLM2-1.7B and TinyLlama-1.1B-Chat-v1.0. The configuration snapshots supply architectural dimensions; synthetic shape frequencies are not claims about deployment traffic. Model weights are not included.

The optional decoder validator follows the declared Llama/Qwen2 dense-decoder architecture and uses [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). Its random weights and independently timed components do not establish accuracy of a pretrained language model or identify the automatically selected SDPA kernel.
