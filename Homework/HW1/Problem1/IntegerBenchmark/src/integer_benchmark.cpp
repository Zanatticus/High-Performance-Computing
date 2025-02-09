// Code taken from
// https://github.com/shaovoon/arithmeticbench/blob/master/CppFloatMulDivBench/CppFloatMulDivBench.cpp
// Modified by: Zander Ingare

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class timer {
	public:
	timer() = default;
	void start(const std::string& text_) {
		text  = text_;
		begin = std::chrono::high_resolution_clock::now();
	}
	void stop() {
		auto end = std::chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto ms  = std::chrono::duration_cast< std::chrono::milliseconds >(dur).count();
		std::cout << std::setw(19) << text << ":" << std::setw(5) << ms << "ms" << std::endl;
	}

	private:
	std::string                                    text;
	std::chrono::high_resolution_clock::time_point begin;
};

std::vector< int64_t > smallIntList;
std::vector< int64_t > bigIntList;

void Init() {
	smallIntList.push_back(158);
	smallIntList.push_back(21);
	smallIntList.push_back(7813);
	smallIntList.push_back(632);
	smallIntList.push_back(87);
	smallIntList.push_back(14);
	smallIntList.push_back(751);
	smallIntList.push_back(201);
	smallIntList.push_back(79);
	smallIntList.push_back(26);

	bigIntList.push_back(158862);
	bigIntList.push_back(78213);
	bigIntList.push_back(425763);
	bigIntList.push_back(412489);
	bigIntList.push_back(852362);
	bigIntList.push_back(23546);
	bigIntList.push_back(145823);
	bigIntList.push_back(352689);
	bigIntList.push_back(558721);
}

int64_t MulBigInt(size_t loop) {
	timer stopwatch;
	stopwatch.start("MulBigInt");

	int64_t result = 0;
	for (size_t k = 0; k < loop; ++k) {
		for (size_t i = 0; i < bigIntList.size(); ++i) {
			for (size_t j = 0; j < bigIntList.size(); ++j) {
				result = bigIntList[i] * bigIntList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

int64_t DivBigInt(size_t loop) {
	timer stopwatch;
	stopwatch.start("DivBigInt");

	int64_t result = 0;
	for (size_t k = 0; k < loop; ++k) {
		for (size_t i = 0; i < bigIntList.size(); ++i) {
			for (size_t j = 0; j < bigIntList.size(); ++j) {
				result = bigIntList[i] / bigIntList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

int64_t MulSmallInt(size_t loop) {
	timer stopwatch;
	stopwatch.start("MulSmallInt");

	int64_t result = 0;
	for (size_t k = 0; k < loop; ++k) {
		for (size_t i = 0; i < smallIntList.size(); ++i) {
			for (size_t j = 0; j < smallIntList.size(); ++j) {
				result = smallIntList[i] * smallIntList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

int64_t DivSmallInt(size_t loop) {
	timer stopwatch;
	stopwatch.start("DivSmallInt");

	int64_t result = 0;
	for (size_t k = 0; k < loop; ++k) {
		for (size_t i = 0; i < smallIntList.size(); ++i) {
			for (size_t j = 0; j < smallIntList.size(); ++j) {
				result = smallIntList[i] / smallIntList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

int64_t AddBigInt(size_t loop) {
	timer stopwatch;
	stopwatch.start("AddBigInt");

	int64_t result = 0;
	for (size_t k = 0; k < loop; ++k) {
		for (size_t i = 0; i < bigIntList.size(); ++i) {
			for (size_t j = 0; j < bigIntList.size(); ++j) {
				result = bigIntList[i] + bigIntList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

int64_t SubBigInt(size_t loop) {
	timer stopwatch;
	stopwatch.start("SubBigInt");

	int64_t result = 0;
	for (size_t k = 0; k < loop; ++k) {
		for (size_t i = 0; i < bigIntList.size(); ++i) {
			for (size_t j = 0; j < bigIntList.size(); ++j) {
				result = bigIntList[i] - bigIntList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

int64_t AddSmallInt(size_t loop) {
	timer stopwatch;
	stopwatch.start("AddSmallInt");

	int64_t result = 0;
	for (size_t k = 0; k < loop; ++k) {
		for (size_t i = 0; i < smallIntList.size(); ++i) {
			for (size_t j = 0; j < smallIntList.size(); ++j) {
				result = smallIntList[i] + smallIntList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

int64_t SubSmallInt(size_t loop) {
	timer stopwatch;
	stopwatch.start("SubSmallInt");

	int64_t result = 0;
	for (size_t k = 0; k < loop; ++k) {
		for (size_t i = 0; i < smallIntList.size(); ++i) {
			for (size_t j = 0; j < smallIntList.size(); ++j) {
				result = smallIntList[i] - smallIntList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

int main() {
	Init();
	size_t loop = 10000000;
	std::cout << "Multiplication and Division Benchmark" << std::endl;
	std::cout << "=====================================" << std::endl;
	MulBigInt(loop);
	DivBigInt(loop);
	MulSmallInt(loop);
	DivSmallInt(loop);
	std::cout << "\nAddition and Subtraction Benchmark" << std::endl;
	std::cout << "==================================" << std::endl;
	AddBigInt(loop);
	SubBigInt(loop);
	AddSmallInt(loop);
	SubSmallInt(loop);

	return 0;
}
