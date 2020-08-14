#pragma once

#define _USE_MATH_DEFINES

#ifdef _LIB
#define DLL_NETWORK __declspec(dllexport)
#else
#define DLL_NETWORK
#endif

#include <cuda.h>
#include <curand.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "intellisense.h"
#include <math.h>
#include <stdint.h>
#include <iostream>
#include "../Control Panel 2/enumerations.cs"
#include "exceptions.h"

__host__ __device__ void applyFunction(float& output, float input, FunctionType type);

typedef uint32_t uint;

bool cuda_assert(const cudaError_t code, const char* const file, const int line);
bool cuda_assert_void(const char* const file, const int line);

#define voidChkErr(...) {										\
	(__VA_ARGS__);												\
	cudaError_t e=cudaGetLastError();							\
	if(e!=cudaSuccess) {										\
        std::cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(e) << "\n";	\
	}															\
}

#define cudav(...)  cuda##__VA_ARGS__; cuda_assert_void(__FILE__, __LINE__);
#define cuda(...)  cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__);

#define overDiv(dividend, divisor) (((dividend) + (divisor) - 1) / (divisor))

#ifdef __INTELLISENSE__
#define KERNELCALL2(function, blocks, threads, ...) voidChkErr(function(__VA_ARGS__))
#define KERNELCALL3(function, blocks, threads, sharedMem, ...) voidChkErr(function(__VA_ARGS__))
#define KERNELCALL4(function, blocks, threads, sharedMem, stream, ...) voidChkErr(function(__VA_ARGS__))
#else
#define KERNELCALL2(function, blocks, threads, ...) voidChkErr(function <<< blocks, threads >>> (__VA_ARGS__))
#define KERNELCALL3(function, blocks, threads, sharedMem, ...) voidChkErr(function <<< blocks, threads, sharedMem >>> (__VA_ARGS__))
#define KERNELCALL4(function, blocks, threads, sharedMem, stream, ...) voidChkErr(function <<< blocks, threads, sharedMem, stream >>> (__VA_ARGS__))
#endif