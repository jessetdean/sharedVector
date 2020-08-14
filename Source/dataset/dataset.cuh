#pragma once

#include "datum.cuh"
#include "batch.cuh"

class DLL_NETWORK Dataset {
public:
	cudaVector<Datum> data;
	Batch batch;

	//Saved derived variables
	uint outputGroups = 0;
	uint outputs = 0;
	uint inputs = 0;
	cudaVector<uint> classSizes;

	//Settings
	bool testSet = false;
	uint batchSize = 1;
	float errorTarget = 0.01f;
	uint recOffset = 0;

	//Use functions

	/*
	 * Check if dataset is empty
	 */
	__host__ __device__ bool isEmpty();

	/*
	 * Get data for a particular instance, batch is read internally
	 */
	__host__ __device__ Datum& getDatum(uint instanceID);

	/*
	 * Get data for a particular instance, batch is read internally
	 */
	__host__ __device__ Datum& operator[](uint instanceID);

	/*
	 * Get input data for a particular instance and input, batch is read internally
	 */
	__host__ __device__ float getInput(uint instanceID, uint inputID);

	/*
	 * Get output data for a particular instance and output, batch is read internally
	 */
	__host__ __device__ float getOutput(uint instanceID, uint outputID);

	/*
	 * Check to see if the given output is a regression type
	 */
	__host__ __device__ bool isRegressionOuput(uint outputID);

	/*
	 * If a given output is a class type, get the corresponding group.
	 * Throws errors if regression type
	 */
	__host__ __device__ uint getClassGroup(uint outputID);

	/*
	 * If a given output is a class type, get the corresponding index within its group.
	 * Throws errors if regression type
	 */
	__host__ __device__ uint getClassIndex(uint outputID);

	/*
	 * Get overall output group ID for statistics. Each regrssion is its own group.
	 */
	__host__ __device__ uint getOutputGroup(uint outputID);
};