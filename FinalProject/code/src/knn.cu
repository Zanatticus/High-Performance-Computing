// Program to compute the k-nearest neighbors of a point in a dataset using CUDA.
// Computes the squared Euclidean distance between a query point and all reference points in the dataset.
// Author: Zander Ingare

#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
