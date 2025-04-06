#ifndef CIFAR_LOADER_H
#define CIFAR_LOADER_H

#include <string>
#include <vector>

class CIFARLoader {
public:
    CIFARLoader();  // loads both train and test internally

    int getImageSize() const;
    int getWidth() const;
    int getHeight() const;
    int getChannels() const;

    int getNumTrainImages() const;
    int getNumTestImages() const;

    const std::vector<float>& getTrainImages() const;
    const std::vector<unsigned char>& getTrainLabels() const;

    const std::vector<float>& getTestImages() const;
    const std::vector<unsigned char>& getTestLabels() const;

    const char* getLabelName(int labelIndex) const;

private:
    int imageSize;
    int width;
    int height;
    int channels;

    std::vector<float> trainImages;
    std::vector<unsigned char> trainLabels;

    std::vector<float> testImages;
    std::vector<unsigned char> testLabels;

    void loadBatchFile(const std::string &filePath, std::vector<float>& imageVec, std::vector<unsigned char>& labelVec);
    void loadTrain();
    void loadTest();

    static constexpr const char* BASE_DIR = "datasets/CIFAR-10/";
    static constexpr const char* TRAIN_BATCHES[5] = {
        "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
        "data_batch_4.bin", "data_batch_5.bin"
    };
    static constexpr const char* TEST_BATCH = "test_batch.bin";

    static constexpr int NUM_CLASSES = 10;
    static const char* LABEL_NAMES[NUM_CLASSES];
};

#endif
