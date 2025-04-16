# GPU-Accelerated KNN Classifier for Image Datasets

This repository contains the implementation of a GPU-accelerated K-Nearest Neighbors (KNN) classifier for image classification using CUDA. The project evaluates the performance of KNN on three popular image datasets: MNIST, CIFAR-10, and STL-10, with a focus on high-performance computing techniques.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [How to Build](#how-to-build)
5. [How to Run](#how-to-run)
6. [Datasets](#datasets)
7. [Performance Metrics](#performance-metrics)
8. [Documentation](#documentation)

---

## Project Overview

This project implements a K-Nearest Neighbors (KNN) classifier accelerated using CUDA for GPU computation. The implementation includes:

- Data loaders for MNIST, CIFAR-10, and STL-10 image datasets
- CUDA-based KNN classifier with distance computation on GPU
- Standard and batched processing approaches for handling large datasets
- Performance metrics collection and analysis
- Image visualization capabilities

## Project Structure

```
FinalProject/
├── README.md                 // This file
├── ProjectDescription.pdf    // Project description document
├── ProjectPresentation.pdf   // Presentation slides
├── ProjectProposal.pdf       // Initial project proposal
├── ProjectReport.pdf         // Final project report
├── code/                     // Main application code
│   ├── Makefile              // Build configuration
│   ├── README.md             // Code-specific documentation
│   ├── clang-format          // Code formatting configuration
│   ├── format-all-files.sh   // Script to format code files
│   ├── build/                // Build artifacts
│   │   └── main              // Compiled executable
│   ├── datasets/             // Image datasets
│   │   ├── CIFAR-10/         // CIFAR-10 dataset files
│   │   ├── MNIST/            // MNIST dataset files
│   │   ├── STL-10/           // STL-10 dataset files
│   │   └── README.md         // Dataset documentation
│   ├── include/              // Header files
│   │   ├── CIFAR-Loader.h    // CIFAR-10 dataset loader
│   │   ├── KNN-Classifier.h  // KNN classifier implementation
│   │   ├── MNIST-Loader.h    // MNIST dataset loader
│   │   └── STL-Loader.h      // STL-10 dataset loader
│   ├── output/               // Generated output files
│   │   ├── cifar_image.ppm   // Sample CIFAR-10 image
│   │   ├── knn_metrics.csv   // Performance metrics gathered on P100 & A100
│   │   ├── mnist_image.ppm   // Sample MNIST image
│   │   └── stl_image.ppm     // Sample STL-10 image
│   └── src/                  // Source files
│       ├── CIFAR-Loader.cpp  // CIFAR-10 dataset loader implementation
│       ├── KNN-Classifier.cu // CUDA implementation of KNN
│       ├── MNIST-Loader.cpp  // MNIST dataset loader implementation
│       ├── STL-Loader.cpp    // STL-10 dataset loader implementation
│       └── main.cpp          // Application entry point
```

## Prerequisites

- NVIDIA GPU with CUDA support (tested on Tesla P100, V100, and A100)
- CUDA Toolkit (compatible with the GPU architecture)
- NVIDIA HPC SDK
- C++ compiler (g++)
- NVIDIA CUDA compiler (nvcc)
- Image datasets (MNIST, CIFAR-10, STL-10)

## How to Build

The project uses a Makefile for building. Navigate to the `code` directory and run:

```bash
make
```

This will compile the code using the NVIDIA HPC SDK and CUDA compiler, targeting the A100 GPU architecture by default. You can modify the Makefile to target different GPU architectures (P100, V100) by changing the architecture flag.

## How to Run

After building the project, you can run the executable with:

```bash
make run
```

Or directly:

```bash
./build/main
```

Modify the makefile directly to target different GPU architectures (P100, V100, A100) by changing the architecture flag used in compilation.

The program will run the KNN classifier on all three datasets (MNIST, CIFAR-10, and STL-10) and output performance metrics to the console and to `output/knn_metrics.csv`.

## Datasets

The project uses three image datasets:

1. **MNIST**: Handwritten digit recognition dataset (28x28 grayscale images)
2. **CIFAR-10**: Object classification dataset (32x32 color images in 10 classes)
3. **STL-10**: Image recognition dataset (96x96 color images in 10 classes)

The datasets should be placed in the `code/datasets/` directory as shown in the project structure.

## Performance Metrics

The application collects and outputs several performance metrics:

- Total execution time (including data loading)
- GPU-only execution time
- Non-GPU overhead time
- GPU memory usage
- Classification accuracy

These metrics are saved to `code/output/knn_metrics.csv` for analysis.

## Documentation

The project includes several documentation files:

- `ProjectDescription.pdf`: Detailed description of the project
- `ProjectProposal.pdf`: Initial project proposal
- `ProjectReport.pdf`: Final project report with results and analysis
- `ProjectPresentation.pdf`: Presentation slides summarizing the project

## Implementation Details

### KNN Classifier

The KNN classifier is implemented in CUDA to leverage GPU parallelism. Key features include:

- Euclidean distance calculation on GPU
- K-nearest neighbors search
- Majority voting for classification
- Support for both standard and batched processing

### Data Loaders

Custom data loaders are implemented for each dataset:

- `MNIST-Loader`: Loads the MNIST handwritten digit dataset
- `CIFAR-Loader`: Loads the CIFAR-10 object classification dataset
- `STL-Loader`: Loads the STL-10 image recognition dataset

Each loader handles the binary format of its respective dataset and provides methods to access the images and labels.

### Performance Optimization

The implementation includes several optimizations:

- Batch processing for large datasets
- CUDA kernel optimizations for distance calculations
- Memory management optimizations to reduce data transfer overhead
- Support for different GPU architectures (P100, V100, A100)

## Code Formatting

This project follows a consistent code formatting style using clang-format. To maintain code consistency across all source files, a formatting script is provided:

```bash
./code/format-all-files.sh
```

The script formats all C/C++/CUDA files (*.c, *.cpp, *.h, *.hpp, *.cu) according to the project's style guidelines. You can run it with the following options:

- `-h, --help`: Show help message
- `-a, --all`: Format files in all subdirectories (default)
- `-n, --name=DIR`: Format files only in the specified directory

Example usage:
```bash
./code/format-all-files.sh --all
./code/format-all-files.sh --name=src
```

The code follows these formatting conventions:
- **Indentation**: Tabs for indentation
- **Braces**: Opening braces on the same line as control statements
- **Line Length**: Maximum of 120 characters
- **Naming Conventions**:
  - Classes/Structs: PascalCase (e.g., `KNNClassifier`)
  - Functions/Methods: camelCase (e.g., `evaluateDataset()`)
  - Variables: camelCase (e.g., `trainImages`)
  - Constants/Macros: UPPER_SNAKE_CASE (e.g., `K_MNIST`)

---

*This project was developed as the final project for the High-Performance Computing course.*
