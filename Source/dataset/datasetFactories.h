#pragma once

#include "dataset.h"
#include <string>
#include <vector>

/*
 * Generate a dataset from a CSV. CSV must have the following format
 * First line: [number of data lines], [labels]...
 * Labels must have their 3 letter type code prepending the name
 * Filename: full directory filename of CSV to read
 */
void DLL_NETWORK readCSV(cuDataset& dataset, std::string filename);

/*
 * Generate a dataset based on numberical functions.
 * It is recommended to use randomInputs and some amount of output noise for training sets to simulate real-world variation.
 * funcs: non-empty vector for the types of functions to be generated. Dataset size will quickly get larger based on this parameter.
 * randomInputs: toggle between randomly sampling inputs between -1 and 1 (recommended for training sets) and evenly distributing them about that range
 * 	  (recommended for testing)
 * outputNoise: sigma for normal distribution of noise to be added to outputs. Value must be positive.
 * 	  Non-zero values recommended only for training sets.
 * samplesPerDimension: number of recursive samples for each dimension. Must be greater than 0.
 * 	  Total dataset size will increase by samplesPerDimension^dimensions
 */
void DLL_NETWORK generateFunction(cuDataset& dataset, std::vector<FunctionType> funcs, bool randomInputs,
	float outputNoise, uint samplesPerDimension);

void DLL_NETWORK generateTimeSeries(cuDataset& dataset, std::vector<SeriesSettings> settings, uint samples, uint inputDelay);

/*
 * Generate a dataset based on the MNIST native files.
 * dataFile: full filename for the MNIST datafile portion.
 * labelFile: full filename for the MNIST labels.
 */
void DLL_NETWORK readMNIST(cuDataset& dataset, std::string dataFile, std::string labelFile);