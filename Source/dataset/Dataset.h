#pragma once

#include "datum.h"
#include "batch.h"
#include "dataset.cuh"

#include <exception>

#pragma warning(disable:4251)

class DLL_NETWORK cuDataset : public cuClass<Dataset> {
public:
	//Core data
	sharedVector<cuDatum> data;
	sharedVector<uint> classSizes;

	//Metadata
	std::vector<std::string> featureLabels;
	std::vector<std::string> outputLabels;
	std::vector<float> minVals;
	std::vector<float> maxVals;
	cuDatum means;
	cuDatum stdDevs;
	bool shouldNormalize = true;

	//Get functions for charting use

	/*
	 * Gets specified output for the entire series
	 */
	std::vector<float> getOutputs(uint outputID);

	/*
	 * Gets specified input for the entire series
	 */
	std::vector<float> getInputs(uint inputID);

	//General use operators

	/*
	 * Get the next valid batch for this dataset.
	 * Optional parameter of last batch for continuation
	 */
	void nextBatch(cuBatch& newBatch, bool recurrent = false, cuBatch* lastBatch = 0);

	/*
	 * Bind a batch to this dataset and send to device. 
	 */
	Dataset& pushBatchAndRender(bool device, cuBatch& batch);

	//Associated dataset generation

	/*
	 * Move data from this dataset into a new one
	 * float percentage: number between 0 and 1 that determines how much of the data to move to the new dataset
	 * bool random: choice to randomly pick data, otherwise it will take data sequentially from the end
	 */
	void split(cuDataset& outSet, float percentage, bool random);

	/*
	 * Copy data from this dataset into another.
	 * float percentage: number between 0 and 1 that determines how much of the data to move to the new dataset
	 * float shift: starting point for the dataset copy
	 */
	void getWindow(cuDataset& outSet, float percentage, float shift);

	/*
	 * Take in another dataset and append it to the end of this one. The input dataset will be destroyed
	 * Dataset set: dataset to be merged
	 */
	void merge(cuDataset& set);

	/*
	 * Normalizes dataset to have a mean 0 and stdDev of 1. Saves mean and stdDev information for later reversion.
	 * Skips if "shouldNormalize" member variable is false.
	 */
	void normalize();

	/*
	 * Returns dataset to its original, non-normalized form.
	 * Should be avoided multiple times, as the floating point math will be lossy.
	 */
	void revertNormalization();

	/*
	 * Reverts normalization for a single output vector, useful for display functions.
	 */
	void revertOutputNorm(std::vector<float>& outputs, uint outputID);

	/*
	 * Reverts normalization for a single input vector, useful for display functions.
	 */
	void revertInputNorm(std::vector<float>& inputs, uint inputID);

	/*
	 * Create a set that fits predetermined resolution for use in the visualizer
	 */
	//void generatePreview();

	//Derived variables
	uint outputGroups();
	uint outputs();
	uint inputs();

	//cuClass overrides
	Dataset& render(bool device = true) override;
	void clearDevice() override;
	void clear() override;
};

#pragma warning(default:4251)