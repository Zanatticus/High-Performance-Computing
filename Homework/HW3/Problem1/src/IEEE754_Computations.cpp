// Program to approximate f(x) = sin(x) using IEEE 754 floating point arithmetic and a Taylor series expansion.
// The program compares the approximation using floats (SP) versus doubles (DP).
// Author: Zander Ingare

#include <cmath>
#include <iomanip>
#include <iostream>

#define NUM_TERMS 10

unsigned long long factorial(int n) {
	if (n == 0)
		return 1;
	unsigned long long result = 1;
	for (int i = 1; i <= n; ++i) {
		result *= i;
	}
	return result;
}

float approximate_sin_float(float x) {
	float sum = 0;

	for (int i = 0; i < NUM_TERMS; ++i) {
		if (i % 2 == 0) {
			sum += std::pow(x, 2 * i + 1) / factorial(2 * i + 1);
		} else {
			sum -= std::pow(x, 2 * i + 1) / factorial(2 * i + 1);
		}
	}

	return sum;
}

double approximate_sin_double(double x) {
	double sum = 0;

	for (int i = 0; i < NUM_TERMS; ++i) {
		if (i % 2 == 0) {
			sum += std::pow(x, 2 * i + 1) / factorial(2 * i + 1);
		} else {
			sum -= std::pow(x, 2 * i + 1) / factorial(2 * i + 1);
		}
	}

	return sum;
}

int main() {
	float  sin_float_0  = approximate_sin_float(0.5);
	double sin_double_0 = approximate_sin_double(0.5);

	float  sin_float_1  = approximate_sin_float(1.0);
	double sin_double_1 = approximate_sin_double(1.0);

	float  sin_float_2  = approximate_sin_float(2.0);
	double sin_double_2 = approximate_sin_double(2.0);

	float  sin_float_3  = approximate_sin_float(3.0);
	double sin_double_3 = approximate_sin_double(3.0);

	std::cout << std::fixed << std::setprecision(30);
	std::cout << "\nUsing " << NUM_TERMS << " terms in the Taylor series expansion." << std::endl;
	std::cout << "\nResults are given in radians:" << std::endl;
	std::cout << "-------------------------------------------------------" << std::endl;
	std::cout << "sin(0.5) using float:  " << sin_float_0 << std::endl;
	std::cout << "sin(0.5) using double: " << sin_double_0 << std::endl;
	std::cout << "Actual value:          " << "0.479425538604203000273287935215" << std::endl;
	std::cout << "-------------------------------------------------------" << std::endl;
	std::cout << "sin(1.0) using float:  " << sin_float_1 << std::endl;
	std::cout << "sin(1.0) using double: " << sin_double_1 << std::endl;
	std::cout << "Actual value:          " << "0.841470984807896506652502321630" << std::endl;
	std::cout << "-------------------------------------------------------" << std::endl;
	std::cout << "sin(2.0) using float:  " << sin_float_2 << std::endl;
	std::cout << "sin(2.0) using double: " << sin_double_2 << std::endl;
	std::cout << "Actual value:          " << "0.909297426825681695396019865911" << std::endl;
	std::cout << "-------------------------------------------------------" << std::endl;
	std::cout << "sin(3.0) using float:  " << sin_float_3 << std::endl;
	std::cout << "sin(3.0) using double: " << sin_double_3 << std::endl;
	std::cout << "Actual value:          " << "0.141120008059867222100744802808" << std::endl;
	std::cout << "-------------------------------------------------------" << std::endl;

	return 0;
}