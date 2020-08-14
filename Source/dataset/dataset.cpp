#include "dataset.h"
#include <random>
#include <time.h>

using namespace std;

static default_random_engine generator = default_random_engine((unsigned int)time(NULL));

vector<float> cuDataset::getOutputs(uint outputID) {
	if (data.size() == 0u || outputID >= (uint)data[0].regressions.size() + (uint)data[0].classes.size())
		throw invalidOutputID();

	vector<float> retVec(data.size());

	bool isRegression = outputID < data[0].regressions.size();
	if (!isRegression) {
		outputID -= (uint)data[0].regressions.size();
	}

	for (uint i = 0; i < (uint)data.size(); i++) {
		if (isRegression) {
			retVec[i] = data[i].regressions[outputID];
		}
		else {
			retVec[i] = (float)data[i].classes[outputID];
		}
	}

	//Reset normalization on this data
	revertOutputNorm(retVec, outputID);

	return retVec;
}

vector<float> cuDataset::getInputs(uint inputID) {
	if (data.size() == 0u || inputID >= (uint)data[0].features.size())
		throw invalidInputID();

	vector<float> retVec(data.size());

	for (uint i = 0; i < (uint)data.size(); i++) {
		retVec[i] = data[i].features[inputID];
	}

	//Reset normalization on this data
	revertInputNorm(retVec, inputID);

	return retVec;
}

void cuDataset::nextBatch(cuBatch& newBatch, bool recurrent, cuBatch* lastBatch) {
	uint recStride = 1;
	if (recurrent) {
		recStride = overDiv((uint)data.size(), core.batchSize);
	}

	//Reuse reset
	newBatch.statistics.setDirty();
	newBatch.selections.setDirty();
	newBatch.classResponses.setDirty();
	//newBatch.responses.setDirty();//Data doesn't matter, only allocation. Not marked transient for debugging purposes

	//newBatch.classResponses.reset();//Not needed so long as responses aren't read back to host
	newBatch.statistics.reset();

	//Allocations
	newBatch.responses.resize(core.batchSize);
	for (int i = 0; i < newBatch.responses.size(); i++)
		newBatch.responses[i].resize(outputs());
	
	newBatch.classResponses.resize(core.batchSize);
	for (int i = 0; i < newBatch.classResponses.size(); i++) {
		newBatch.classResponses[i].resize(classSizes.size());
	}
	newBatch.statistics.resize(outputGroups());
	newBatch.selections.resize(core.batchSize);

	//Get pointers to next set
	if ((uint)data.size() == 0u)
		return;
	if (lastBatch != 0 && recurrent && lastBatch->selections.size() > 0 && lastBatch->selections[0] < recStride - 1) {
		//If this isn't the first iteration, we just need to increment
		for (int num = 0; num < newBatch.selections.size(); num++) {
			newBatch.selections[num] = (uint)lastBatch->selections[num] + 1;//Handle out of bounds at kernel level
		}
	}
	else {
		if (lastBatch != 0  && !recurrent)
			newBatch.lastIndex = lastBatch->lastIndex;
		else
			newBatch.lastIndex = 0;

		//Get integer pointer to selected datasets
		if (core.testSet || recurrent) {
			for (uint num = 0; num < core.batchSize; num++, newBatch.lastIndex += recStride) {
				if (newBatch.lastIndex >= data.size()) {
					newBatch.selections.resize(num);
					newBatch.lastIndex = 0;
					break;
				}

				//select the next sample
				newBatch.selections[num] = newBatch.lastIndex;
			}
		}
		else {
			if (newBatch.lastIndex + core.batchSize > (uint)data.size() - 1) {
				newBatch.selections.resize((uint)data.size() - newBatch.lastIndex);
				newBatch.lastIndex = 0;
			}
			else {
				newBatch.lastIndex += core.batchSize;
			}

			//Random pool selection
			uniform_int_distribution<uint> distribution(0, (uint)data.size() - 1);
			for (uint num = 0; num < newBatch.selections.size(); num++) {
				//select a sample from the pool
				newBatch.selections[num] = distribution(generator);
			}
		}
	}

	return;
}

Dataset& cuDataset::pushBatchAndRender(bool device, cuBatch& batch) {
	core.batch = batch.render(device);
	return render(device);
}

void cuDataset::split(cuDataset& outSet, float percentage, bool random) {
	if (percentage <= 0.0f || percentage >= 1.0f)
		throw invalidSplitParameter();

	//Clear
	outSet.clear();

	//Copy metadata
	outSet.featureLabels = featureLabels;
	outSet.outputLabels = outputLabels;
	outSet.classSizes = classSizes;

	//Settings
	outSet.core.batchSize = core.batchSize;

	//Randomly select datapoints to move sets
	int oldSize = (int)(data.size() * (1 - percentage));
	if (random) {
		while (data.size() > oldSize) {
			uniform_int_distribution<int>distribution(0, (int)data.size() - 1);
			int index = distribution(generator);
			outSet.data.push_back(data[index]);
			data.erase(data.begin() + index);
		}
	}
	else {
		for (int i = oldSize; i < data.size(); i++) {
			outSet.data.push_back(data[i]);
		}
		data.resize(oldSize);
	}

	//Copy over normalization information
	outSet.means = means;
	outSet.stdDevs = stdDevs;
	outSet.shouldNormalize = shouldNormalize;

	//Set new offset to be old size
	outSet.core.recOffset = oldSize;
}

void cuDataset::getWindow(cuDataset& outSet, float percentage, float shift) {
	if (shift < 0.0f || shift >= 1.0f ||
		percentage <= 0.0f || percentage > 1.0f ||
		shift + percentage > 1.0f)
		throw invalidWindowParameter();

	//Clear
	outSet.clear();

	//Copy metadata
	outSet.featureLabels = featureLabels;
	outSet.outputLabels = outputLabels;
	outSet.classSizes = classSizes;

	//Settings
	outSet.core.batchSize = core.batchSize;

	//Randomly select datapoints to move sets
	uint offset = (uint)(shift * data.size());
	for (int i = 0; i < (int)(data.size() * percentage); i++) {
		outSet.data.push_back(data[i + offset]);
	}

	//Copy over normalization information
	outSet.means = means;
	outSet.stdDevs = stdDevs;
	outSet.shouldNormalize = shouldNormalize;

	//Set new offset to be old size
	outSet.core.recOffset = 0;
}

void cuDataset::merge(cuDataset& set) {
	//Reset both datasets
	revertNormalization();
	set.revertNormalization();

	data.reserve(data.size() + set.data.size());
	data.insert(data.end(), set.data.begin(), set.data.end());

	set.data.clear();

	//Remake normalization for merged set
	normalize();
}

void cuDataset::normalize() {
	if (!shouldNormalize)
		return;

	//Inputs
	for (uint i = 0; i < inputs(); i++) {
		//Get mean
		float mean = 0.0f;
		{
			float min = FLT_MAX;
			float max = -FLT_MAX;
			for (uint j = 0; j < data.size(); j++) {
				float val = data[j].features[i];
				mean += val;
				if (val > max)
					max = val;
				if (val < min)
					min = val;
			}
			minVals.push_back(min);
			maxVals.push_back(max);
		}
		mean /= data.size();
		means.features.push_back(mean);

		//Get std dev
		float stdDev = 0.0f;
		for (uint j = 0; j < data.size(); j++) {
			stdDev += pow(data[j].features[i] - mean, 2.0f);
		}
		stdDev = sqrt(stdDev / data.size());
		stdDevs.features.push_back(stdDev);

		//Normalize
		for (uint j = 0; j < data.size(); j++) {
			data[j].features[i] = (data[j].features[i] - mean) / stdDev;
		}
	}

	//Outputs
	for (uint i = 0; i < (uint)data[0].regressions.size(); i++) {
		//Get mean
		float mean = 0.0f;
		for (uint j = 0; j < data.size(); j++) {
			mean += data[j].regressions[i];
		}
		mean /= data.size();
		means.regressions.push_back(mean);

		//Get std dev
		float stdDev = 0.0f;
		for (uint j = 0; j < data.size(); j++) {
			stdDev += pow(data[j].regressions[i] - mean, 2.0f);
		}
		stdDev = sqrt(stdDev / data.size());
		stdDevs.regressions.push_back(stdDev);

		//Normalize
		for (uint j = 0; j < data.size(); j++) {
			data[j].regressions[i] = (data[j].regressions[i] - mean) / stdDev;
		}
	}
}

void cuDataset::revertNormalization() {
	//Inputs
	if (means.features.size() == 0)
		return;
	for (uint i = 0; i < inputs(); i++) {
		float mean = means.features[i];
		float stdDev = stdDevs.features[i];

		//Normalize
		for (uint j = 0; j < data.size(); j++) {
			data[j].features[i] = data[j].features[i] * stdDev + mean;
		}
	}

	//Outputs
	if (means.regressions.size() == 0)
		return;
	for (uint i = 0; i < (uint)data[0].regressions.size(); i++) {
		float mean = means.regressions[i];
		float stdDev = stdDevs.regressions[i];

		//DeNormalize
		for (uint j = 0; j < data.size(); j++) {
			data[j].regressions[i] = data[j].regressions[i] * stdDev + mean;
		}
	}

	means = {};
	stdDevs = {};
}

void cuDataset::revertOutputNorm(vector<float>& outputs, uint outputID) {
	if (means.regressions.size() == 0)
		return;

	float mean = means.regressions[outputID];
	float stdDev = stdDevs.regressions[outputID];

	//DeNormalize
	for (uint j = 0; j < outputs.size(); j++) {
		outputs[j] = outputs[j] * stdDev + mean;
	}
}

void cuDataset::revertInputNorm(vector<float>& inputs, uint inputID) {
	if (means.features.size() == 0)
		return;

	float mean = means.features[inputID];
	float stdDev = stdDevs.features[inputID];

	//DeNormalize
	for (uint j = 0; j < inputs.size(); j++) {
		inputs[j] = inputs[j] * stdDev + mean;
	}
}

uint cuDataset::outputGroups() {
	if (core.outputGroups == 0) {
		if (data.size() > 0) {
			core.outputGroups = (uint)data[0].regressions.size() + (uint)data[0].classes.size();
		}
	}
	return core.outputGroups;
}

uint cuDataset::outputs() {
	if (core.outputs == 0) {
		if (data.size() > 0) {
			core.outputs = (uint)data[0].regressions.size();
			for (uint i = 0; i < (uint)classSizes.size(); i++)
				core.outputs += classSizes[i];
		}
	}
	return core.outputs;
}

uint cuDataset::inputs() {
	if (core.inputs == 0) {
		if (data.size() > 0) {
			core.inputs = (uint)data[0].features.size();
		}
	}
	return core.inputs;
}

void cuDataset::clearDevice() {
	data.clearDevice();
	classSizes.clearDevice();
}

void cuDataset::clear() {
	//Clear data holding vectors
	data.clear();
	classSizes.clear();
	featureLabels.clear();
	outputLabels.clear();
	minVals.clear();
	maxVals.clear();
	means = {};
	stdDevs = {};

	core.outputGroups = 0;
	core.outputs = 0;
	core.inputs = 0;

	//Flag for remake
	data.setDirty();
	classSizes.setDirty();
}

Dataset& cuDataset::render(bool device) {
	core.data = data.render(device);
	if (core.batch.selections.size() == 0)
		throw noBatchException();
	core.classSizes = classSizes.render(device);

	//Saved derived variables
	core.inputs = inputs();
	core.outputs = outputs();
	core.outputGroups = outputGroups();

	return core;
}