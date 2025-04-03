#include "MNIST-Loader.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

MNISTLoader::MNISTLoader(const std::string &imageFilePath, const std::string &labelFilePath)
    : numImages(0), numRows(0), numCols(0)
{
    loadImages(imageFilePath);
    loadLabels(labelFilePath);
}

int MNISTLoader::getNumImages() const {
    return numImages;
}

int MNISTLoader::getImageSize() const {
    return numRows * numCols;
}

const std::vector<float>& MNISTLoader::getImages() const {
    return images;
}

const std::vector<unsigned char>& MNISTLoader::getLabels() const {
    return labels;
}

int MNISTLoader::readBigEndianInt(FILE *fp) {
    unsigned char bytes[4];
    if (fread(bytes, sizeof(unsigned char), 4, fp) != 4) {
        throw std::runtime_error("Failed to read 4 bytes from file");
    }
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

void MNISTLoader::loadImages(const std::string &filePath) {
    FILE *fp = fopen(filePath.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open image file: " + filePath);

    int magic = readBigEndianInt(fp);
    if (magic != 2051) throw std::runtime_error("Invalid image file format: " + filePath);

    numImages = readBigEndianInt(fp);
    numRows = readBigEndianInt(fp);
    numCols = readBigEndianInt(fp);
    int imageSize = numRows * numCols;

    images.resize(numImages * imageSize);
    unsigned char *temp = new unsigned char[numImages * imageSize];
    fread(temp, sizeof(unsigned char), numImages * imageSize, fp);
    fclose(fp);

    for (int i = 0; i < numImages * imageSize; ++i) {
        images[i] = temp[i] / 255.0f; // normalize
    }

    delete[] temp;
}

void MNISTLoader::loadLabels(const std::string &filePath) {
    FILE *fp = fopen(filePath.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open label file: " + filePath);

    int magic = readBigEndianInt(fp);
    if (magic != 2049) throw std::runtime_error("Invalid label file format: " + filePath);

    int count = readBigEndianInt(fp);
    if (count != numImages) throw std::runtime_error("Label count does not match image count");

    labels.resize(count);
    fread(labels.data(), sizeof(unsigned char), count, fp);
    fclose(fp);
}
