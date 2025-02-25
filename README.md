# CUDA-Accelerated Matrix Multiplication

This project demonstrates the performance difference between CPU and GPU (CUDA) implementations of complex matrix multiplication. It uses `numpy` for CPU-based computation and `cupy` (a GPU-accelerated library) for GPU-based computation. The project highlights the significant speedup achieved by leveraging GPU parallelism.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

Matrix multiplication is a fundamental operation in many scientific computing and machine learning tasks. While CPUs can handle these computations, GPUs are designed to perform parallel operations much faster. This project compares the performance of matrix multiplication on a CPU (using `numpy`) and a GPU (using `cupy`).

The project generates two random complex matrices of size `1000x1000` and performs matrix multiplication on both the CPU and GPU over 100 epochs. It then calculates the average time taken by each and the speedup achieved by the GPU.

## Requirements

To run this project, you need the following:

- Python 3.x
- `numpy` (for CPU computation)
- `cupy` (for GPU computation)
- A CUDA-capable GPU with the appropriate drivers installed

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/cuda-matrix-multiplication.git
   cd cuda-matrix-multiplication
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install numpy cupy
   ```

## Usage

Run the script to compare CPU and GPU performance:
```bash
python matrix_multiplication.py
```

The script will:
1. Generate two random complex matrices of size `1000x1000`.
2. Perform matrix multiplication on the CPU and GPU over 100 epochs.
3. Calculate and display the average time taken by the CPU and GPU, as well as the speedup achieved by the GPU.

## Results

After running the script, you will see output similar to the following:
```
Average time taken by CPU over 100 epochs: 0.123456 seconds
Average time taken by GPU over 100 epochs: 0.012345 seconds
Speedup: 10.00x
```

This indicates that the GPU implementation is significantly faster than the CPU implementation for this task.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

