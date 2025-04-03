#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <string>
#include <vector>

class MNISTLoader {
public:
    MNISTLoader(const std::string &imageFilePath, const std::string &labelFilePath);

    int getNumImages() const;
    int getImageSize() const;

    const std::vector<float>& getImages() const;
    const std::vector<unsigned char>& getLabels() const;

private:
    int numImages;
    int numRows;
    int numCols;

    std::vector<float> images;           // normalized [0,1]
    std::vector<unsigned char> labels;

    int readBigEndianInt(FILE *fp);
    void loadImages(const std::string &filePath);
    void loadLabels(const std::string &filePath);
};

#endif
