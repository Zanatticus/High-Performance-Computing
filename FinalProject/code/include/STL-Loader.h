#ifndef STL_LOADER_H
#define STL_LOADER_H

#include <string>
#include <vector>

class STLLoader {
public:
	STLLoader();

	int getImageSize() const;
	int getWidth() const;
	int getHeight() const;
	int getChannels() const;

	int getNumTrainImages() const;
	int getNumTestImages() const;

	const std::vector<float>&         getTrainImages() const;
	const std::vector<unsigned char>& getTrainLabels() const;
	const std::vector<float>&         getTestImages() const;
	const std::vector<unsigned char>& getTestLabels() const;

	void writeImageToPPM(const std::vector<float>& image, int index, const std::string& filename) const;

private:
	static constexpr const char* BASE_DIR    = "datasets/STL-10/";
	static constexpr const char* TRAIN_IMAGE = "train_X.bin";
	static constexpr const char* TRAIN_LABEL = "train_y.bin";
	static constexpr const char* TEST_IMAGE  = "test_X.bin";
	static constexpr const char* TEST_LABEL  = "test_y.bin";

	int width, height, channels, imageSize;

	std::vector<float>         trainImages;
	std::vector<unsigned char> trainLabels;
	std::vector<float>         testImages;
	std::vector<unsigned char> testLabels;

	void loadImageFile(const std::string& path, std::vector<float>& imageVec);
	void loadLabelFile(const std::string& path, std::vector<unsigned char>& labelVec);
};

#endif
