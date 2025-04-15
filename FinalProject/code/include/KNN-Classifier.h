#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#include <cuda_runtime.h>
#include <string>
#include <vector>

class KNNClassifier {
public:
	KNNClassifier(const std::vector<float>&         trainImages,
	              const std::vector<unsigned char>& trainLabels,
	              const std::vector<float>&         testImages,
	              const std::vector<unsigned char>& testLabels,
	              const std::string&                datasetName,
	              int                               k        = 5,
				  bool 							    useBatchMode = false,
	              int                               deviceId = 0);

	~KNNClassifier();

	void train();
	void trainBatched(); // New method for batch processing setup

	unsigned char predict(int imageIndex);

	float evaluateDataset();
	float evaluateDatasetBatched(); // New method for batch evaluation

	double getGpuExecutionTime() const;
	float getGpuMemoryUsage() const;
	std::string getGpuType() const;
	int getGpuCount() const;

private:
	// Device memory
	float*         d_trainImages;
	unsigned char* d_trainLabels;
	float*         d_testImage;    // For standard approach: single test image
	float*         d_testImages;   // For batched approach: all test images
	float*         d_distances;
	int*           d_indices;
	unsigned char* d_predictedLabel;

	// Host memory
	std::vector<float>         h_trainImages;
	std::vector<unsigned char> h_trainLabels;
	std::vector<float>         h_testImages;
	std::vector<unsigned char> h_testLabels;
	std::string                datasetName;

	int k_neighbors;
	int numTrainImages;
	int numTestImages;
	int imageSize;
	int deviceId;

	double gpuExecutionTime;
	float  gpuMemoryUsage;
	bool   useBatchMode = false; // Flag for batch processing mode

	void allocateDeviceMemory();
	void allocateDeviceMemoryBatched(); // For batch processing
	void freeDeviceMemory();
	void sortDistancesAndFindMajority();
	void computeDistances();
};

#endif // KNN_CLASSIFIER_H