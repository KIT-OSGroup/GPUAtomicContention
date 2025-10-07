# Benchmark Suite for "[Are Your GPU Atomics Secretly Contending?](https://dl.acm.org/doi/10.1145/3764860.3768338)"

This repository contains all necessary files and instructions to run the benchmark suite presented in our PLOS '25 paper [Are Your GPU Atomics Secretly Contending?](https://dl.acm.org/doi/10.1145/3764860.3768338).

## Note: Draft status
The repository is currently in **draft status**.
While all benchmarks presented in the paper are already added,
some extensions are still missing.
The repository will be completed by the time the workshop starts on Monday, October 13 2025,
at which point this note will be removed.

## Dependencies

**Hardware Requirements:**

- An AMD GPU using a RDNA1, RDNA2, or RDNA3 instruction set
- An Nvidia GPU supporting at least Compute Capability 7.0 (Volta or newer)

**Operating System:** Linux

**Software Requirements:**

- Docker (with NVIDIA Container Toolkit) or podman
- CMake
- Make
- Python
- Matplotlib

## Configuration

Copy `config.sh.example` to `config.sh` and modify it for your setup.
Refer to the table below for explanation on the settings.

|              **Setting**             | **Type** | **Description**                                                                                                                                                                                                |
|:------------------------------------:|:--------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|               `NVIDIA`               |  Boolean | Enable benchmarking for NVIDIA GPUs.                                                                                                                                                                           |
|           `NVIDIA_DEVICES`           |  String  | NVIDIA GPUs to benchmark as a comma separated list of device UUIDs. Use `all` only with a single NVIDIA GPU installed.                                                                                         |
|           `NVIDIA_FOLDERS`           |  String  | Folder names to store benchmark data and to use in plots as a comma separated list. Must match `NVIDIA_DEVICES` in length.                                                                                     |
|                 `AMD`                |  Boolean | Enable benchmarking for AMD GPUs.                                                                                                                                                                              |
|             `AMD_DEVICES`            |  String  | AMD GPUs to benchmark as a comma separated list of DRI devices (e.g. `/dev/dri/renderD128`). Use `/dev/dri` only with a single AMD GPU installed.                                                              |
|             `AMD_FOLDERS`            |  String  | Folder names to store benchmark data and to use in plots as a comma separated list. Must match `AMD_DEVICES` in length.                                                                                        |
|            `BM_CONTENTION`           |  Boolean | Enable the atomic contention benchmark.                                                                                                                                                                        |
|         `BM_CONTENTION_RUNS`         |  Integer | Number of runs for the atomic contention benchmarks.                                                                                                                                                           |
|     `BM_CONTENTION_BASELINE_RUNS`    |  Integer | Number of runs for the atomic contention baseline benchmarks.                                                                                                                                                  |
|         `BM_CONTENTION_MODES`        |  String  | Modes to use in the atomic contention benchmark as a comma separated list. Allowed values: `add_acquire`, `add_seq_cst`, `add_relaxed`, `add_acquire_sync`, `add_seq_cst_sync`, `add_relaxed_sync`, `spinlock` |
|        `BM_CONTENTION_SCOPES`        |  String  | Memory Scopes to use in the atomic contention benchmark as a comma separated list. Allowed values: `block`, `device`, `system`                                                                                 |
|         `BM_CONTENTION_GRIDS`        |  String  | Grids to use in the atomic contention benchmark as a comma separated list. Allowed values: `small`, `large`                                                                                                    |
|    `BM_CONTENTION_TRANSPOSE_MODES`   |  String  | Transpose modes to use in the atomic contention benchmark as a comma separated list. Allowed values: `direct`, `scattered`                                                                                     |
|      `BM_CONTENTION_NO_BASELINE`     |  Boolean | Disable baseline benchmarks for the atomic contention benchmark.                                                                                                                                               |
|  `BM_CONTENTION_NO_VARYING_THREADS`  |  Boolean | Disable varying thread benchmarks for the atomic contention benchmark.                                                                                                                                         |
|  `BM_CONTENTION_NO_VARYING_STRIDES`  |  Boolean | Disable varying memory stride benchmarks for the atomic contention benchmark.                                                                                                                                  |
| `BM_CONTENTION_NO_OFFSETTED_STRIDES` |  Boolean | Disable non-power-of-two memory stride benchmarks for the atomic contention benchmark.                                                                                                                         |

## Running

Execute `make` in the root directory to build, run, and evaluate the benchmarks for all configured GPUs.
Note, the evaluation scripts only generate the plots used in the paper
and require running all atomic contention benchmarks at least for `add_relaxed` operations on `device` scope.

Other useful commands include:

- `make run` to run build and run the benchmarks without evaluation.
- `make run-amd` or `make run-nvidia` to run only on AMD or NVIDIA GPUs, respectively.
- `make enter-amd` or `make enter-nvidia` to get a shell in the respective docker environment with all GPUs passed through for the respective vendor.
- `make plot` to evaluate without rerunning the benchmarks.

## Benchmarks and Tests

The benchmark suite includes the following programs:

- `bm_contention`, a benchmark evaluating atomic performance under contention and different memory access patterns.
- `bm_atomics`, a benchmark evaluating specific atomic stores and a spinlock implemented using either an _Atomic Or_ or an _Atomic Compare and Swap_ operation.
- `test_globaltimer`, a utility program to measure the rate of the globaltimer we use for measurement.
- `test_gpu_cpu_atomics`, a program to test PCIe atomics between the GPU and CPU.
- `test_cross_gpu_atomics`, a program to test PCIe atomics between two or more GPUs. This program uses a shared file for synchronization and the actual test.
- `test_barriers`, a program to test different grid layouts for maximum occupancy using a simple barrier implementation.

## Citation

```bibtex
@inproceedings{10.1145/3764860.3768338,
author = {Maucher, Peter and Djerfi, Nick and Kittner, Lennard and Werling, Lukas and Bellosa, Frank},
title = {Are Your GPU Atomics Secretly Contending?},
year = {2025},
isbn = {9798400722257},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3764860.3768338},
doi = {10.1145/3764860.3768338},
abstract = {GPU applications use atomic operations to coordinate data access in highly parallel code. However, relying on previous experiences and due to limited documentation, programmers resort to guidelines instead of concrete metrics to limit potential performance influences.In this paper, we introduce a GPU memory-subsystem microbenchmark suite for analyzing GPU atomic operations. Based on the benchmark results, we discuss two particular guidelines, namely: "use only one thread per warp to access an atomic" and "place two atomic variables on different cache lines to avoid contention." We demonstrate where these guidelines are effective and where actual hardware behavior diverges.},
booktitle = {Proceedings of the 13th Workshop on Programming Languages and Operating Systems},
pages = {84–92},
numpages = {9},
keywords = {Atomic Contention, Atomic Operations, GPU, Microbenchmarks, Synchronization},
location = {Seoul, Republic of Korea},
series = {PLOS '25}
}```
