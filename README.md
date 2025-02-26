# ML-Compiler-Optimization-for-Edge-Devices


# Compiler-based Optimization for ML Models on Heterogeneous Multi-Core Systems
This repository showcases a compiler optimization framework for improving the performance of machine learning models on heterogeneous multi-core systems, including CPU, GPU, and custom accelerators like Qualcomm’s Neural Signal Processor.

## Getting Started
To explore the project, clone the repository and follow the setup instructions below to build and run the optimization framework.

## Features
* Optimizes ML models for multi-core and heterogeneous hardware architectures
* Reduces latency and improves throughput by using tensor parallelism and operator fusion
* Enhances memory efficiency with custom memory allocation strategies
* Supports deployment across mobile, edge, and cloud platforms

## Technical Details
The compiler optimization framework includes the following key techniques:
* **Tensor Parallelism:** Splits large tensors across multiple cores to enhance processing efficiency.
* **Operator Fusion:** Combines multiple tensor operations into one kernel to minimize inter-device communication.
* **Memory Optimization:** Applies memory tiling and custom allocation strategies to optimize memory usage.
* **Autotuning:** Uses reinforcement learning and grid search for optimal model parameters and system settings.
* **MLIR & LLVM Backend:** Targets Qualcomm’s Adreno GPUs and other hardware accelerators for improved performance.

## Project Structure
The project consists of the following components:
* **Compiler:** Custom compiler to optimize ML workloads for heterogeneous systems.
* **Optimization Algorithms:** Implementations of tensor parallelism, operator fusion, and memory optimization.
* **Autotuning Framework:** Automated tuning of model parameters for hardware-specific optimizations.
* **Benchmarking:** Profiling and benchmarking scripts to evaluate performance improvements.

## Dependencies
* **MLIR** (for compiler framework)
* **LLVM** (for backend support)
* **TensorFlow/PyTorch** (for ML model integration)
* **Python** (for autotuning and profiling scripts)

## Contributing
Contributions are welcome! To contribute, fork this repository, make your changes, and submit a pull request.

