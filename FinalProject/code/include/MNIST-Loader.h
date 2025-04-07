#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <string>
#include <vector>

class MNISTLoader {
public:
    MNISTLoader();  // ← no arguments

    int getNumTrainImages() const;
    int getNumTestImages() const;
    int getImageSize() const;

    const std::vector<float>& getTrainImages() const;
    const std::vector<unsigned char>& getTrainLabels() const;
    const std::vector<float>& getTestImages() const;
    const std::vector<unsigned char>& getTestLabels() const;

    void writeImageToPPM(const std::vector<float>& image, int index, const std::string& filename) const;

private:
    int numRows;
    int numCols;

    std::vector<float> trainImages;
    std::vector<unsigned char> trainLabels;
    std::vector<float> testImages;
    std::vector<unsigned char> testLabels;

    int readBigEndianInt(FILE *fp);
    void loadImagesAndLabels(const std::string &imagePath, const std::string &labelPath,
                             std::vector<float> &imageVec, std::vector<unsigned char> &labelVec);

    // Internally defined constants
    static constexpr const char* BASE_DIR = "datasets/MNIST/";
    static constexpr const char* TRAIN_IMAGE_FILE = "train-images.idx3-ubyte";
    static constexpr const char* TRAIN_LABEL_FILE = "train-labels.idx1-ubyte";
    static constexpr const char* TEST_IMAGE_FILE  = "t10k-images.idx3-ubyte";
    static constexpr const char* TEST_LABEL_FILE  = "t10k-labels.idx1-ubyte";
};

#endif
