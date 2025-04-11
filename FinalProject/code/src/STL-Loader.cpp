#include "STL-Loader.h"

#include <fstream>
#include <stdexcept>

STLLoader::STLLoader() : width(96), height(96), channels(3) {
	imageSize = width * height * channels;
	loadImageFile(std::string(BASE_DIR) + TRAIN_IMAGE, trainImages);
	loadLabelFile(std::string(BASE_DIR) + TRAIN_LABEL, trainLabels);
	loadImageFile(std::string(BASE_DIR) + TEST_IMAGE, testImages);
	loadLabelFile(std::string(BASE_DIR) + TEST_LABEL, testLabels);
}

void STLLoader::loadImageFile(const std::string& path, std::vector<float>& imageVec) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("Cannot open STL image file: " + path);

	file.seekg(0, std::ios::end);
	size_t fileSize = file.tellg();
	file.seekg(0, std::ios::beg);

	size_t numImages = fileSize / imageSize;
	imageVec.resize(numImages * imageSize);

	std::vector<unsigned char> buffer(fileSize);
	file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
	file.close();

	for (size_t i = 0; i < fileSize; ++i)
		imageVec[i] = buffer[i] / 255.0f;
}

void STLLoader::loadLabelFile(const std::string& path, std::vector<unsigned char>& labelVec) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("Cannot open STL label file: " + path);

	file.seekg(0, std::ios::end);
	size_t fileSize = file.tellg();
	file.seekg(0, std::ios::beg);

	labelVec.resize(fileSize);
	file.read(reinterpret_cast<char*>(labelVec.data()), fileSize);
	file.close();
}

int STLLoader::getImageSize() const {
	return imageSize;
}
int STLLoader::getWidth() const {
	return width;
}
int STLLoader::getHeight() const {
	return height;
}
int STLLoader::getChannels() const {
	return channels;
}

int STLLoader::getNumTrainImages() const {
	return trainLabels.size();
}
int STLLoader::getNumTestImages() const {
	return testLabels.size();
}

const std::vector<float>& STLLoader::getTrainImages() const {
	return trainImages;
}
const std::vector<unsigned char>& STLLoader::getTrainLabels() const {
	return trainLabels;
}
const std::vector<float>& STLLoader::getTestImages() const {
	return testImages;
}
const std::vector<unsigned char>& STLLoader::getTestLabels() const {
	return testLabels;
}

void STLLoader::writeImageToPPM(const std::vector<float>& image, int index, const std::string& filename) const {
	std::ofstream out(filename);
	if (!out.is_open())
		throw std::runtime_error("Cannot open PPM file for writing: " + filename);

	out << "P3\n" << width << " " << height << "\n255\n";
	int offset = index * imageSize;

	// STL-10 stores channels in column-major order
	for (int row = 0; row < height; ++row) {
		for (int col = 0; col < width; ++col) {
			int idx_col_major = col * height + row;
			int r             = static_cast<int>(image[offset + idx_col_major] * 255.0f);
			int g             = static_cast<int>(image[offset + width * height + idx_col_major] * 255.0f);
			int b             = static_cast<int>(image[offset + 2 * width * height + idx_col_major] * 255.0f);
			out << r << " " << g << " " << b << "\n";
		}
	}
	out.close();
}
