# High-Performance Computing Final Project Code






foir the STL dataset, i have to manually set the size of the shared memory for each block since the size of the image (108 KB) is too big for the default (48 KB) shared memory size. This is overridden at runtime for V100, P100, and A100










```
=== Testing KNN on MNIST dataset ===
MNIST Train Size: 60000
MNIST Test Size: 10000
KNN: Loaded MNIST training data with 60000 images of size 784
MNIST Test Image Actual Label: 7
MNIST Test Image Predicted Label: 7
KNN: Processed 1000/10000 test images. Current accuracy: 96.1%
KNN: Processed 2000/10000 test images. Current accuracy: 95.65%
KNN: Processed 3000/10000 test images. Current accuracy: 95.3667%
KNN: Processed 4000/10000 test images. Current accuracy: 95.5%
KNN: Processed 5000/10000 test images. Current accuracy: 95.46%
KNN: Processed 6000/10000 test images. Current accuracy: 95.8667%
KNN: Processed 7000/10000 test images. Current accuracy: 96.2%
KNN: Processed 8000/10000 test images. Current accuracy: 96.575%
KNN: Processed 9000/10000 test images. Current accuracy: 96.8333%
KNN: Processed 10000/10000 test images. Current accuracy: 96.88%
KNN: Final accuracy: 96.88%
KNN: Total evaluation time: 47.9018 seconds
KNN: GPU memory usage: 182 MB

=== Testing KNN on CIFAR dataset ===
CIFAR Train Size: 50000
CIFAR Test Size: 10000
KNN: Loaded CIFAR training data with 50000 images of size 3072
CIFAR Test Image Actual Label: 3 (cat)
CIFAR Test Image Predicted Label: 2 (bird)
KNN: Processed 1000/10000 test images. Current accuracy: 34.7%
KNN: Processed 2000/10000 test images. Current accuracy: 33.75%
KNN: Processed 3000/10000 test images. Current accuracy: 33.9667%
KNN: Processed 4000/10000 test images. Current accuracy: 33.875%
KNN: Processed 5000/10000 test images. Current accuracy: 33.9%
KNN: Processed 6000/10000 test images. Current accuracy: 33.6333%
KNN: Processed 7000/10000 test images. Current accuracy: 33.9%
KNN: Processed 8000/10000 test images. Current accuracy: 34.0125%
KNN: Processed 9000/10000 test images. Current accuracy: 34.0333%
KNN: Processed 10000/10000 test images. Current accuracy: 33.98%
KNN: Final accuracy: 33.98%
KNN: Total evaluation time: 157.163 seconds
KNN: GPU memory usage: 586 MB
```