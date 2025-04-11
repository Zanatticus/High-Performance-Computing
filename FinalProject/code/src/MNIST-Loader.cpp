#include "MNIST-Loader.h"

#include <fstream>
#include <stdexcept>

MNISTLoader::MNISTLoader() {
	loadImagesAndLabels(
	    std::string(BASE_DIR) + TRAIN_IMAGE_FILE, std::string(BASE_DIR) + TRAIN_LABEL_FILE, trainImages, trainLabels);

	loadImagesAndLabels(
	    std::string(BASE_DIR) + TEST_IMAGE_FILE, std::string(BASE_DIR) + TEST_LABEL_FILE, testImages, testLabels);
}

int MNISTLoader::readBigEndianInt(FILE *fp) {
	unsigned char bytes[4];
	fread(bytes, sizeof(unsigned char), 4, fp);
	return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

void MNISTLoader::loadImagesAndLabels(const std::string          &imagePath,
                                      const std::string          &labelPath,
                                      std::vector<float>         &imageVec,
                                      std::vector<unsigned char> &labelVec) {
	FILE *fp = fopen(imagePath.c_str(), "rb");
	if (!fp)
		throw std::runtime_error("Cannot open image file: " + imagePath);

	int magic = readBigEndianInt(fp);
	if (magic != 2051)
		throw std::runtime_error("Invalid image file format: " + imagePath);

	int numImages = readBigEndianInt(fp);
	numRows       = readBigEndianInt(fp);
	numCols       = readBigEndianInt(fp);
	int imageSize = numRows * numCols;

	imageVec.resize(numImages * imageSize);
	unsigned char *temp = new unsigned char[numImages * imageSize];
	fread(temp, sizeof(unsigned char), numImages * imageSize, fp);
	fclose(fp);

	for (int i = 0; i < numImages * imageSize; ++i) {
		imageVec[i] = temp[i] / 255.0f;
	}
	delete[] temp;

	// Labels
	fp = fopen(labelPath.c_str(), "rb");
	if (!fp)
		throw std::runtime_error("Cannot open label file: " + labelPath);
	magic = readBigEndianInt(fp);
	if (magic != 2049)
		throw std::runtime_error("Invalid label file format: " + labelPath);

	int numLabels = readBigEndianInt(fp);
	if (numLabels != numImages)
		throw std::runtime_error("Image/label count mismatch");

	labelVec.resize(numLabels);
	fread(labelVec.data(), sizeof(unsigned char), numLabels, fp);
	fclose(fp);
}

int MNISTLoader::getNumTrainImages() const {
	return trainLabels.size();
}
int MNISTLoader::getNumTestImages() const {
	return testLabels.size();
}
int MNISTLoader::getImageSize() const {
	return numRows * numCols;
}

const std::vector<float> &MNISTLoader::getTrainImages() const {
	return trainImages;
}
const std::vector<unsigned char> &MNISTLoader::getTrainLabels() const {
	return trainLabels;
}
const std::vector<float> &MNISTLoader::getTestImages() const {
	return testImages;
}
const std::vector<unsigned char> &MNISTLoader::getTestLabels() const {
	return testLabels;
}

void MNISTLoader::writeImageToPPM(const std::vector<float> &image, int index, const std::string &filename) const {
	std::ofstream out(filename);
	if (!out.is_open())
		throw std::runtime_error("Cannot open PPM file for writing: " + filename);

	out << "P3\n" << numCols << " " << numRows << "\n255\n";
	int offset = index * numCols * numRows;
	for (int i = 0; i < numCols * numRows; ++i) {
		int gray = static_cast<int>(image[offset + i] * 255.0f);
		out << gray << " " << gray << " " << gray << "\n";
	}

	out.close();
}