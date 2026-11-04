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

The external documents explain measurement concepts. Actual claims about this experiment come from the [published dataset](../results/README.md), [all-model metrics](../results/summary/model_metrics.csv), [quality records](../results/summary/quality_and_audits.csv), and executable source. Official documentation does not establish this project's measured accuracy or validate its generalization claims.
