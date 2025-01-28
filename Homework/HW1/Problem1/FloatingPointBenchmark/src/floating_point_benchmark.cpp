// Code taken from https://github.com/shaovoon/arithmeticbench/blob/master/CppFloatMulDivBench/CppFloatMulDivBench.cpp
// Modified by: Zander Ingare

#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <cmath>
#include <cassert>
#include <sstream>
#include <cstdlib>
#include <chrono>

class timer
{
public:
	timer() = default;
	void start(const std::string& text_)
	{
		text = text_;
		begin = std::chrono::high_resolution_clock::now();
	}
	void stop()
	{
		auto end = std::chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << std::setw(19) << text << ":" << std::setw(5) << ms << "ms" << std::endl;
	}

private:
	std::string text;
	std::chrono::high_resolution_clock::time_point begin;
};

std::vector<double> smallDoubleList;
std::vector<double> bigDoubleList;

void Init()
{
	smallDoubleList.push_back(158.0);
	smallDoubleList.push_back(21.0);
	smallDoubleList.push_back(7813.0);
	smallDoubleList.push_back(632.0);
	smallDoubleList.push_back(87.0);
	smallDoubleList.push_back(14.0);
	smallDoubleList.push_back(751.0);
	smallDoubleList.push_back(201.0);
	smallDoubleList.push_back(79.0);
	smallDoubleList.push_back(26.0);

	bigDoubleList.push_back(158862.0);
	bigDoubleList.push_back(78213.0);
	bigDoubleList.push_back(425763.0);
	bigDoubleList.push_back(412489.0);
	bigDoubleList.push_back(852362.0);
	bigDoubleList.push_back(23546.0);
	bigDoubleList.push_back(145823.0);
	bigDoubleList.push_back(352689.0);
	bigDoubleList.push_back(558721.0);
}

double MulBigDouble(size_t loop)
{
	timer stopwatch;
	stopwatch.start("MulBigDouble");

	double result = 0.0;
	for (size_t k = 0; k < loop; ++k)
	{
		for (size_t i = 0; i < bigDoubleList.size(); ++i)
		{
			for (size_t j = 0; j < bigDoubleList.size(); ++j)
			{
				result = bigDoubleList[i] * bigDoubleList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

double DivBigDouble(size_t loop)
{
	timer stopwatch;
	stopwatch.start("DivBigDouble");

	double result = 0.0;
	for (size_t k = 0; k < loop; ++k)
	{
		for (size_t i = 0; i < bigDoubleList.size(); ++i)
		{
			for (size_t j = 0; j < bigDoubleList.size(); ++j)
			{
				result = bigDoubleList[i] / bigDoubleList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

double MulSmallDouble(size_t loop)
{
	timer stopwatch;
	stopwatch.start("MulSmallDouble");

	double result = 0.0;
	for (size_t k = 0; k < loop; ++k)
	{
		for (size_t i = 0; i < smallDoubleList.size(); ++i)
		{
			for (size_t j = 0; j < smallDoubleList.size(); ++j)
			{
				result = smallDoubleList[i] * smallDoubleList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

double DivSmallDouble(size_t loop)
{
	timer stopwatch;
	stopwatch.start("DivSmallDouble");

	double result = 0.0;
	for (size_t k = 0; k < loop; ++k)
	{
		for (size_t i = 0; i < smallDoubleList.size(); ++i)
		{
			for (size_t j = 0; j < smallDoubleList.size(); ++j)
			{
				result = smallDoubleList[i] / smallDoubleList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

double AddBigDouble(size_t loop)
{
	timer stopwatch;
	stopwatch.start("AddBigDouble");

	double result = 0.0;
	for (size_t k = 0; k < loop; ++k)
	{
		for (size_t i = 0; i < bigDoubleList.size(); ++i)
		{
			for (size_t j = 0; j < bigDoubleList.size(); ++j)
			{
				result = bigDoubleList[i] + bigDoubleList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

double SubBigDouble(size_t loop)
{
	timer stopwatch;
	stopwatch.start("SubBigDouble");

	double result = 0.0;
	for (size_t k = 0; k < loop; ++k)
	{
		for (size_t i = 0; i < bigDoubleList.size(); ++i)
		{
			for (size_t j = 0; j < bigDoubleList.size(); ++j)
			{
				result = bigDoubleList[i] - bigDoubleList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

double AddSmallDouble(size_t loop)
{
	timer stopwatch;
	stopwatch.start("AddSmallDouble");

	double result = 0.0;
	for (size_t k = 0; k < loop; ++k)
	{
		for (size_t i = 0; i < smallDoubleList.size(); ++i)
		{
			for (size_t j = 0; j < smallDoubleList.size(); ++j)
			{
				result = smallDoubleList[i] + smallDoubleList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}

double SubSmallDouble(size_t loop)
{
	timer stopwatch;
	stopwatch.start("SubSmallDouble");

	double result = 0.0;
	for (size_t k = 0; k < loop; ++k)
	{
		for (size_t i = 0; i < smallDoubleList.size(); ++i)
		{
			for (size_t j = 0; j < smallDoubleList.size(); ++j)
			{
				result = smallDoubleList[i] - smallDoubleList[j];
			}
		}
	}
	stopwatch.stop();

	return result;
}


int main()
{
	Init();
	size_t loop = 10000000;
	std::cout << "Multiplication and Division Benchmark" << std::endl;
	std::cout << "=====================================" << std::endl;
	MulBigDouble(loop);
	DivBigDouble(loop);
	MulSmallDouble(loop);
	DivSmallDouble(loop);
	std::cout << "\nAddition and Subtraction Benchmark" << std::endl;
	std::cout << "==================================" << std::endl;
	AddBigDouble(loop);
	SubBigDouble(loop);
	AddSmallDouble(loop);
	SubSmallDouble(loop);

	return 0;
}


