#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#include <vector>
#include <string>
#include <cuda_runtime.h>

class KNNClassifier {
public:
    // Constructor with k value and optional device ID
    KNNClassifier(int k = 5, int deviceId = 0);
    ~KNNClassifier();

    // Train with either dataset type
    void trainMNIST(const std::vector<float>& trainImages, 
                   const std::vector<unsigned char>& trainLabels);
    
    void trainCIFAR(const std::vector<float>& trainImages, 
                   const std::vector<unsigned char>& trainLabels);
    
    // Predict single image
    unsigned char predict(const std::vector<float>& image, int imageIndex);
    
    // Predict multiple images and return accuracy
    float evaluateAccuracy(const std::vector<float>& testImages, 
                          const std::vector<unsigned char>& testLabels);
    
    // Getters for performance metrics
    double getLastExecutionTime() const;
    float getGpuMemoryUsage() const;
    
    // Helper functions
    void setK(int newK);
    int getK() const;

private:
    // Device data
    float* d_trainImages;
    unsigned char* d_trainLabels;
    float* d_testImage;
    float* d_distances;
    int* d_indices;
    unsigned char* d_predictedLabel;
    
    // Host data
    int k;
    int numTrainImages;
    int imageSize;
    int deviceId;
    
    // Performance metrics
    double executionTime;
    float gpuMemoryUsage;
    
    // Internal functions
    void allocateDeviceMemory(int numImages, int imgSize);
    void freeDeviceMemory();
    void sortDistancesAndFindMajority();
    
    // CUDA kernel launcher functions (implemented in .cu file)
    void computeDistances(int numTestImages);
};

#endif // KNN_CLASSIFIER_H