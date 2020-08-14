#pragma once

#include "cudaIncludes.h"
#include <exception>

#pragma warning(disable:4251)

/*
Template wrapper class for index in sharedVectors for compiler identification
*/
template<typename T>
class DLL_NETWORK vectorReference {
public:
	uint ID = 0;

	/*
	 * Needed for empty vector initializations
	 */
	vectorReference() {}

	/*
	 * Standard initialization, pass through ID
	 */
	vectorReference(uint ID) : ID(ID) {}

	/*
	 * Check for equality by checking internal uint
	 */
	friend bool operator==(const vectorReference& l, const vectorReference& r) { return l.ID == r.ID; }

	/*
	 * Accessor returns internal int
	 */
	__host__ __device__ operator uint() { return ID; }
};

template<typename T>
class DLL_NETWORK cudaVector {
public:
	uint _size;
	T* d_ptr;

	/*
	 * Vector-style accessor for array with boundary checking
	 */
	__host__ __device__ T& operator [](uint index) { 
#ifdef __CUDA_ARCH__
		if (index >= _size)
			assert(false);
#else
		if (index >= _size)
			throw outOfBounds();
#endif // !__NVCC__

		return d_ptr[index]; 
	}
	/*
	 * Function to return the read only size varible
	 */
	__host__ __device__ uint size() {
		return _size;
	}
};

#pragma warning(default:4251)