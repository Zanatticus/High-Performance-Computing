#ifndef CIFAR_LOADER_H
#define CIFAR_LOADER_H

#include <string>
#include <vector>

/**
 * @class CIFARLoader
 * @brief Handles loading and management of the CIFAR-10 dataset
 *
 * This class provides functionality to load the CIFAR-10 dataset, access training and testing
 * images and labels, and write image data to PPM format for visualization.
 */
class CIFARLoader {
	public:
	/**
	 * @brief Constructor - loads both training and test datasets automatically
	 *
	 * Initializes the loader and reads all CIFAR-10 data from the predefined directory.
	 */
	CIFARLoader();

	/**
	 * @brief Get the total size of each image
	 * @return Size of each image in number of elements (width * height * channels)
	 */
	int getImageSize() const;

	/**
	 * @brief Get the width of each image
	 * @return Width in pixels
	 */
	int getWidth() const;

	/**
	 * @brief Get the height of each image
	 * @return Height in pixels
	 */
	int getHeight() const;

	/**
	 * @brief Get the number of color channels per image
	 * @return Number of channels (typically 3 for RGB)
	 */
	int getChannels() const;

	/**
	 * @brief Get the number of images in the training set
	 * @return Number of training images
	 */
	int getNumTrainImages() const;

	/**
	 * @brief Get the number of images in the test set
	 * @return Number of test images
	 */
	int getNumTestImages() const;

	/**
	 * @brief Get all training images as a flattened vector of normalized pixel values
	 * @return Const reference to the vector containing all training images
	 */
	const std::vector<float>& getTrainImages() const;

	/**
	 * @brief Get all training labels
	 * @return Const reference to the vector containing all training labels
	 */
	const std::vector<unsigned char>& getTrainLabels() const;

	/**
	 * @brief Get all test images as a flattened vector of normalized pixel values
	 * @return Const reference to the vector containing all test images
	 */
	const std::vector<float>& getTestImages() const;

	/**
	 * @brief Get all test labels
	 * @return Const reference to the vector containing all test labels
	 */
	const std::vector<unsigned char>& getTestLabels() const;

	/**
	 * @brief Get the text name for a class label
	 * @param labelIndex Index of the class label (0-9)
	 * @return String representing the class name
	 */
	const char* getLabelName(int labelIndex) const;

	/**
	 * @brief Writes a single image to a PPM file for visualization
	 * @param image Vector containing all images
	 * @param index Index of the specific image to write
	 * @param filename Output filename for the PPM file
	 */
	void writeImageToPPM(const std::vector<float>& image,
	                     int                       index,
	                     const std::string&        filename) const;

	private:
	int imageSize;   ///< Total size of each image (width * height * channels)
	int width;       ///< Width of each image
	int height;      ///< Height of each image
	int channels;    ///< Number of color channels

	std::vector<float>         trainImages;   ///< Training images (normalized)
	std::vector<unsigned char> trainLabels;   ///< Training labels
	std::vector<float>         testImages;    ///< Test images (normalized)
	std::vector<unsigned char> testLabels;    ///< Test labels

	/**
	 * @brief Load data from a single CIFAR-10 binary batch file
	 * @param filePath Path to the batch file
	 * @param imageVec Output vector for the image data
	 * @param labelVec Output vector for the label data
	 */
	void loadBatchFile(const std::string&          filePath,
	                   std::vector<float>&         imageVec,
	                   std::vector<unsigned char>& labelVec);

	/**
	 * @brief Load all training batch files
	 */
	void loadTrain();

	/**
	 * @brief Load the test batch file
	 */
	void loadTest();

	static constexpr const char* BASE_DIR         = "datasets/CIFAR-10/";
	static constexpr const char* TRAIN_BATCHES[5] = {"data_batch_1.bin",
	                                                 "data_batch_2.bin",
	                                                 "data_batch_3.bin",
	                                                 "data_batch_4.bin",
	                                                 "data_batch_5.bin"};
	static constexpr const char* TEST_BATCH       = "test_batch.bin";

	static constexpr int NUM_CLASSES = 10;
	static const char*   LABEL_NAMES[NUM_CLASSES];   ///< String names for each class
};

#endif