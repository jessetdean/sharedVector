#include "dataset.cuh"

__host__ __device__ bool Dataset::isEmpty() { 
	return data.size() == 0; 
}

__host__ __device__ Datum& Dataset::getDatum(uint index) { 
	return data[batch.selections[index]];
}

__host__ __device__ Datum& Dataset::operator[](uint index) { 
	return getDatum(index); 
}

__host__ __device__ float Dataset::getInput(uint index, uint inputID) {
	return getDatum(index).features[inputID];
}

__host__ __device__ float Dataset::getOutput(uint index, uint outputID) {
	Datum thisDatum = getDatum(index);
	if (outputID < thisDatum.regressions.size()) {
		return thisDatum.regressions[outputID];
	}
	outputID -= thisDatum.regressions.size();

	for (uint i = 0; i < (uint)classSizes.size(); i++) {
		if (outputID < classSizes[i])
			return thisDatum.classes[i] == outputID ? 1.0f : 0.0f;
		outputID -= classSizes[i];
	}

#ifdef __CUDA_ARCH__
	assert(false);
#else
	throw invalidOutputID();
#endif // !__NVCC__
	return 0.0f;
}

__host__ __device__ bool Dataset::isRegressionOuput(uint outputID) {
	return outputID < getDatum(0).regressions.size();
}

__host__ __device__ uint Dataset::getClassGroup(uint outputID) {
	Datum thisDatum = getDatum(0);
	if (outputID < thisDatum.regressions.size()) {
#ifdef __CUDA_ARCH__
		assert(false);
#else
		throw classRequestAsRegression();
#endif // !__NVCC__
		return 0;
	}
	outputID -= thisDatum.regressions.size();

	for (uint i = 0; i < (uint)classSizes.size(); i++) {
		if (outputID < classSizes[i])
			return i;
		outputID -= classSizes[i];
	}

#ifdef __CUDA_ARCH__
	assert(false);
#else
	throw invalidOutputID();
#endif // !__NVCC__
	return 0;
}

__host__ __device__ uint Dataset::getClassIndex(uint outputID) {
	Datum thisDatum = getDatum(0);
	if (outputID < thisDatum.regressions.size()) {
#ifdef __CUDA_ARCH__
		assert(false);
#else
		throw classRequestAsRegression();
#endif // !__NVCC__
		return 0;
	}
	outputID -= thisDatum.regressions.size();

	for (uint i = 0; i < (uint)classSizes.size(); i++) {
		if (outputID < classSizes[i])
			return outputID;
		outputID -= classSizes[i];
	}

#ifdef __CUDA_ARCH__
	assert(false);
#else
	throw invalidOutputID();
#endif // !__NVCC__
	return 0;
}

__host__ __device__ uint Dataset::getOutputGroup(uint outputID) {
	Datum thisDatum = getDatum(0);
	if (outputID < thisDatum.regressions.size()) {
		return outputID;
	}
	outputID -= thisDatum.regressions.size();

	for (uint i = 0; i < (uint)classSizes.size(); i++) {
		if (outputID < classSizes[i])
			return i + thisDatum.regressions.size();
		outputID -= classSizes[i];
	}

#ifdef __CUDA_ARCH__
	assert(false);
#else
	throw invalidOutputID();
#endif // !__NVCC__
	return 0;
}