#include "cudaVectorTests.h"

__global__ void sumTo0Cu(cudaVector<int> vec) {
	if(threadIdx.x > 0)
		atomicAdd(&vec[0], vec[threadIdx.x]);
}

void sumTo0(cudaVector<int>& vec) {
	KERNELCALL2(sumTo0Cu, 1, vec.size(), vec);
}

__global__ void incrementCu(cudaVector<int> vec) {
	vec[threadIdx.x]++;
}

void increment(cudaVector<int>& vec) {
	KERNELCALL2(incrementCu, 1, vec.size(), vec);
}

__global__ void overflowCu(cudaVector<int> vec) {
	vec[vec.size()] = 0;
}

void overflow(cudaVector<int>& vec) {
	KERNELCALL2(overflowCu, 1, 1, vec);
}

__global__ void getSizeCu(cudaVector<int> vec, uint* size) {
	*size = vec.size();
}

uint getSize(cudaVector<int>& vec) {
	uint* size, size_h;
	cudaMalloc(&size, sizeof(uint));
	KERNELCALL2(getSizeCu, 1, 1, vec, size);
	cudaMemcpy(&size_h, size, sizeof(uint), cudaMemcpyDeviceToHost);
	return size_h;
}