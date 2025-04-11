#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <string>
#include <vector>

/**
 * @class MNISTLoader
 * @brief Handles loading and management of the MNIST handwritten digits dataset
 *
 * This class provides functionality to load the MNIST dataset, access training and testing images
 * and labels, and write image data to PPM format for visualization.
 */
class MNISTLoader {
	public:
	/**
	 * @brief Constructor - loads both training and test datasets automatically
	 *
	 * Initializes the loader and reads all MNIST data from the predefined directory.
	 */
	MNISTLoader();

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
	 * @brief Get the size (in pixels) of a single image
	 * @return Total number of pixels in each image (numRows * numCols)
	 */
	int getImageSize() const;

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
	 * @brief Writes a single image to a PPM file for visualization
	 * @param image Vector containing all images
	 * @param index Index of the specific image to write
	 * @param filename Output filename for the PPM file
	 */
	void writeImageToPPM(const std::vector<float>& image, int index, const std::string& filename) const;

	private:
	int numRows;   ///< Number of rows in each image
	int numCols;   ///< Number of columns in each image

	std::vector<float>         trainImages;   ///< Training images (normalized)
	std::vector<unsigned char> trainLabels;   ///< Training labels
	std::vector<float>         testImages;    ///< Test images (normalized)
	std::vector<unsigned char> testLabels;    ///< Test labels

	/**
	 * @brief Read a 4-byte big-endian integer from a file
	 * @param fp File pointer to read from
	 * @return The integer value read
	 */
	int readBigEndianInt(FILE* fp);

	/**
	 * @brief Load images and labels from IDX format files
	 * @param imagePath Path to the image file
	 * @param labelPath Path to the label file
	 * @param imageVec Output vector for the image data
	 * @param labelVec Output vector for the label data
	 */
	void loadImagesAndLabels(const std::string&          imagePath,
	                         const std::string&          labelPath,
	                         std::vector<float>&         imageVec,
	                         std::vector<unsigned char>& labelVec);

	// Internally defined constants
	static constexpr const char* BASE_DIR         = "datasets/MNIST/";
	static constexpr const char* TRAIN_IMAGE_FILE = "train-images.idx3-ubyte";
	static constexpr const char* TRAIN_LABEL_FILE = "train-labels.idx1-ubyte";
	static constexpr const char* TEST_IMAGE_FILE  = "t10k-images.idx3-ubyte";
	static constexpr const char* TEST_LABEL_FILE  = "t10k-labels.idx1-ubyte";
};

#endif