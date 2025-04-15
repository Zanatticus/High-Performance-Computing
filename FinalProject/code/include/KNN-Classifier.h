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
	              int                               deviceId = 0);

	~KNNClassifier();

	void train();

	unsigned char predict(int imageIndex);

	float evaluateDataset();

	double getGpuExecutionTime() const;

	float getGpuMemoryUsage() const;

	std::string getGpuType() const;

	int getGpuCount() const;

private:
	// Device memory
	float*         d_trainImages;
	unsigned char* d_trainLabels;
	float*         d_testImage;
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
	bool   configurableSharedMemory = true;

	void allocateDeviceMemory();
	void freeDeviceMemory();
	void sortDistancesAndFindMajority();
	void computeDistances();
};

#endif   // KNN_CLASSIFIER_H