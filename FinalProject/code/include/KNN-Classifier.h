#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#include <cuda_runtime.h>
#include <string>
#include <vector>

/**
 * @class KNNClassifier
 * @brief CUDA-accelerated K-Nearest Neighbors classifier
 *
 * This class implements a K-Nearest Neighbors classifier optimized using CUDA
 * for efficient processing of image data from datasets like MNIST and CIFAR-10.
 */
class KNNClassifier {
	public:
	/**
	 * @brief Constructor for the KNN classifier
	 * @param k Number of nearest neighbors to consider (default: 5)
	 * @param deviceId CUDA device ID to use (default: 0)
	 */
	KNNClassifier(int k = 5, int deviceId = 0);

	/**
	 * @brief Destructor - frees allocated CUDA memory
	 */
	~KNNClassifier();

	/**
	 * @brief Common training method for all datasets
	 * @param trainImages Vector of flattened, normalized training images
	 * @param trainLabels Vector of corresponding training labels
	 * @param datasetName Name of the dataset for logging
	 */
	void train(const std::vector<float>&         trainImages,
	           const std::vector<unsigned char>& trainLabels,
	           const std::string&                datasetName);

	/**
	 * @brief Predict the class of a single image
	 * @param image Vector containing all test images
	 * @param imageIndex Index of the specific image to classify
	 * @return Predicted class label
	 */
	unsigned char predict(const std::vector<float>& image, int imageIndex);

	/**
	 * @brief Predict classes for a batch of images
	 * @param images Vector containing all test images
	 * @param startIndex Starting index of the batch
	 * @param batchSize Number of images to process in this batch
	 * @param predictions Vector to store the predictions (must be pre-allocated)
	 */
	 void predictBatch(const std::vector<float>&   images,
		int                         startIndex,
		int                         batchSize,
		std::vector<unsigned char>& predictions);

	/**
	 * @brief Evaluate the classifier accuracy on a test set
	 * @param testImages Vector of flattened, normalized test images
	 * @param testLabels Vector of corresponding test labels
	 * @return Classification accuracy as a percentage (0-100)
	 */
	float evaluateDataset(const std::vector<float>&         testImages,
	                      const std::vector<unsigned char>& testLabels);
	
	/**
	 * @brief Evaluate the classifier accuracy on a test set with batched processing
	 * @param testImages Vector of flattened, normalized test images
	 * @param testLabels Vector of corresponding test labels
	 * @return Classification accuracy as a percentage (0-100)
	 */
	float evaluateDatasetBatched(const std::vector<float>&         testImages,
		const std::vector<unsigned char>& testLabels);

	/**
	 * @brief Get the GPU execution time of the classifier
	 * @return GPU execution time in seconds
	 */
	double getGpuExecutionTime() const;

	/**
	 * @brief Get the GPU memory usage of the classifier
	 * @return Memory usage in MB
	 */
	float getGpuMemoryUsage() const;

	/**
	 * @brief Get the GPU device name/type
	 * @return String containing the GPU device name
	 */
	std::string getGpuType() const;

	/**
	 * @brief Get the number of available CUDA devices
	 * @return Number of CUDA devices
	 */
	int getGpuCount() const;

	private:
	// Device data
	float*         d_trainImages;      ///< Training images in device memory
	unsigned char* d_trainLabels;      ///< Training labels in device memory
	float*         d_testImage;        ///< Current test image in device memory
	float*         d_distances;        ///< Calculated distances in device memory
	int*           d_indices;          ///< Sorted indices in device memory
	unsigned char* d_predictedLabel;   ///< Predicted label in device memory

	// Host data
	int k;                ///< Number of neighbors to consider
	int numTrainImages;   ///< Number of training images
	int imageSize;        ///< Size of each image in elements
	int deviceId;         ///< CUDA device ID

	// Performance metrics
	double gpuExecutionTime;    ///< Time for GPU execution in seconds
	float  gpuMemoryUsage;      ///< GPU memory usage in MB

	/**
	 * @brief Allocate device memory for training and test data
	 * @param numImages Number of training images
	 * @param imgSize Size of each image in elements
	 */
	void allocateDeviceMemory(int numImages, int imgSize);

	/**
	 * @brief Free all allocated device memory
	 */
	void freeDeviceMemory();

	/**
	 * @brief Sort distances and find the majority label among k nearest neighbors
	 */
	void sortDistancesAndFindMajority();

	/**
	 * @brief Compute distances between test images and all training images using CUDA
	 * @param numTestImages Number of test images to process
	 */
	void computeDistances(int numTestImages);
};

#endif   // KNN_CLASSIFIER_H