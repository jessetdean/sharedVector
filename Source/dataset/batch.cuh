#pragma once

#include "cudaVector.h"

struct Response {
	int answer = -1;
	float maxResponse = -FLT_MAX;
	float denominator = 0.0f;
};

struct Statistics {
	float accuracy = 0.0f;
	float totalError = 0.0f;
};

class DLL_NETWORK Batch {
public:
	cudaVector<uint> selections;
	cudaVector<Statistics> statistics;
	cudaVector<cudaVector<Response> > classResponses;
	cudaVector<cudaVector<float> > responses;
};