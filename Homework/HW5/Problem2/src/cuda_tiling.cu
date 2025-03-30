// A program that optimizes a "nearest neighbor" or "stencil" computation using CUDA on a single GPU.
// Author: Zander Ingare

#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <chrono>

#define n 32

int main() {
    float a[n][n][n], b[n][n][n];
    for (i=1; i<n-1; i++)
    for (j=1; j<n-1; j++)
    for (k=1; k<n-1; k++) {
    a[i][j][k]=0.75*(b[i-1][j][k]+b[i+1][j][k]+b[i][j-1][k]
    + b[i][j+1][k]+b[i][j][k-1]+b[i][j][k+1]);
    }
    return 0;
}
