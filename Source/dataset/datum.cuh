#pragma once

#include "cudaVector.h"

class Datum {
public:
	cudaVector<float> regressions;
	cudaVector<uint> classes;
	cudaVector<float> features;
};