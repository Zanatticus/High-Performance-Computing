#include "CIFAR-Loader.h"

#include <fstream>
#include <stdexcept>

const char* CIFARLoader::LABEL_NAMES[10] = {
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

CIFARLoader::CIFARLoader() : width(32), height(32), channels(3) {
	imageSize = width * height * channels;
	loadTrain();
	loadTest();
}

void CIFARLoader::loadTrain() {
	for (const char* filename : TRAIN_BATCHES) {
		loadBatchFile(std::string(BASE_DIR) + filename, trainImages, trainLabels);
	}
}

void CIFARLoader::loadTest() {
	loadBatchFile(std::string(BASE_DIR) + TEST_BATCH, testImages, testLabels);
}

void CIFARLoader::loadBatchFile(const std::string&          filePath,
                                std::vector<float>&         imageVec,
                                std::vector<unsigned char>& labelVec) {
	std::ifstream file(filePath, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("Cannot open CIFAR-10 file: " + filePath);

	const int                  recordSize = 1 + imageSize;
	std::vector<unsigned char> buffer(recordSize);

	while (file.read(reinterpret_cast<char*>(buffer.data()), recordSize)) {
		labelVec.push_back(buffer[0]);
		for (int i = 1; i < recordSize; ++i) {
			imageVec.push_back(buffer[i] / 255.0f);
		}
	}

	file.close();
}

int CIFARLoader::getImageSize() const {
	return imageSize;
}
int CIFARLoader::getWidth() const {
	return width;
}
int CIFARLoader::getHeight() const {
	return height;
}
int CIFARLoader::getChannels() const {
	return channels;
}

int CIFARLoader::getNumTrainImages() const {
	return trainLabels.size();
}
int CIFARLoader::getNumTestImages() const {
	return testLabels.size();
}

const std::vector<float>& CIFARLoader::getTrainImages() const {
	return trainImages;
}
const std::vector<unsigned char>& CIFARLoader::getTrainLabels() const {
	return trainLabels;
}

const std::vector<float>& CIFARLoader::getTestImages() const {
	return testImages;
}
const std::vector<unsigned char>& CIFARLoader::getTestLabels() const {
	return testLabels;
}

const char* CIFARLoader::getLabelName(int labelIndex) const {
	if (labelIndex < 0 || labelIndex >= NUM_CLASSES)
		return "unknown";
	return LABEL_NAMES[labelIndex];
}

void CIFARLoader::writeImageToPPM(const std::vector<float>& image,
                                  int                       index,
                                  const std::string&        filename) const {
	std::ofstream out(filename);
	if (!out.is_open())
		throw std::runtime_error("Cannot open PPM file for writing: " + filename);

	out << "P3\n" << width << " " << height << "\n255\n";
	int offset = index * imageSize;

	for (int i = 0; i < width * height; ++i) {
		int r = static_cast<int>(image[offset + i] * 255.0f);
		int g = static_cast<int>(image[offset + width * height + i] * 255.0f);
		int b = static_cast<int>(image[offset + 2 * width * height + i] * 255.0f);
		out << r << " " << g << " " << b << "\n";
	}

	out.close();
}
