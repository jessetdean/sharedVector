#pragma once

#ifdef __INTELLISENSE__
int atomicCAS(int*, int, int);
int __float_as_int(float);
float __int_as_float(int);
void __syncthreads();
float atomicAdd(float*, float);
int atomicAdd(int*, int);
int atomicOr(int*, int);
float __shfl(float, int);
float __shfl_down(float, int);
cudaError_t cudaMemcpyToSymbolAsync(Network, Network*, size_t, size_t);
cudaError_t cudaMemcpyToSymbolAsync(cuDataset, cuDataset*, size_t, size_t);
cudaError_t cudaMemcpyToSymbolAsync(cuBatch, cuBatch*, size_t, size_t);
void surf2Dwrite(float, cudaSurfaceObject_t, float, float);
#endif