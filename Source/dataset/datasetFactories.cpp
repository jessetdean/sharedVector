#include "datasetFactories.h"
#include <fstream>
#include <sstream>
#include <queue>
#include <unordered_map>
#include <mutex>
#include "exceptions.h"
#include "math.h"

using namespace std;

mutex mapMtx;
mutex maxMtx;

static default_random_engine generator = default_random_engine((unsigned int)time(NULL));

///////////////////Read CSV

//Set populate methods
static void readCSVThread(vector<string>& lines, ifstream& file) {
	string line;
	for (auto iter = lines.begin(); iter != lines.end(); ++iter) {
		if (!getline(file, line))
			return;
		*iter = line;
	}
}

//Seperate line by line parse threads to allow disk reads to be uninterupted 
static void parseCSVThread(cuDatum& dataRow, string& line, unordered_map<int, unordered_map<string, int> >& classMap, 
	vector<DType>& types, vector<DType>& inOuts) {
	//Parse data into dataset rows
	string cell;
		//TODO: handle outputs not in behind inputs.
	stringstream LS(line);
	int i = 0;
	int errorIter = 0;
	while (getline(LS, cell, ','))
	{
		sharedVector<float>* target = 0;
		switch (inOuts[i]) {
		case DType::Feature:
			target = &dataRow.features;
			break;
		case DType::Regression:
			target = &dataRow.regressions;
			break;
		case DType::Classification:
			break;
		case DType::Error:
			continue;
			{
				//TODO
				/*float value = abs(stof(cell));
				errorRow[0][errorIter] = value;

				errorIter++;*/
			}
			break;
		default:
			break;
		}

		switch (types[i]) {
		case DType::Char:
			//TODO
			break;
		case DType::Int:
		{
			int value = stoi(cell);
			target->push_back(*(float*)(&value));

			if(inOuts[i] == DType::Feature) {
				unique_lock<mutex> lck(mapMtx);
			}
		}
		break;
		case DType::Float: {
			float value = stof(cell);
			target->push_back(value);

			if (inOuts[i] == DType::Feature) {
				unique_lock<mutex> lck(mapMtx);
			}
		}
					break;
		case DType::String:
		{
			uint test;

			{
				unique_lock<mutex> lck(mapMtx);

				//Check for existing classification
				test = classMap[i][cell];
				if (test == 0) {
					//Add new one
					test = (uint)classMap[i].size();
					classMap[i][cell] = test;
				}
			}

			//Save index
			dataRow.classes.push_back(--test);
		}
		break;
		}
		i++;
	}
}

void readCSV(cuDataset& dataset, std::string filename) {
	ifstream file(filename);
	if (!file.is_open())
		throw fileNotFoundException();

	dataset.clear();

	//Get dictionary
	vector<string> result;
	string line;
	getline(file, line);

	stringstream lineStream(line);
	string cell;
	while (getline(lineStream, cell, ','))
	{
		result.push_back(cell);
	}

	vector<DType> types;
	vector<DType> inOut;
	int errors = 0;
	unordered_map<int, unordered_map<string, int> > classMap;
	for (int i = 1; i < result.size(); i++) {
		string typeIn = result[i].substr(0, 3);
		DType type;
		if (strcmp(typeIn.c_str(), "CHR") == 0) {
			types.push_back(DType::Char);
			type = DType::Char;
		}
		else if (strcmp(typeIn.c_str(), "INT") == 0) {
			types.push_back(DType::Int);
			type = DType::Int;
		}
		else if (strcmp(typeIn.c_str(), "FLT") == 0) {
			types.push_back(DType::Float);
			type = DType::Float;
		}
		else if (strcmp(typeIn.c_str(), "STR") == 0) {
			//Add new entry to unordered_map<TKey, TValue>
			unordered_map<string, int> newMap;
			classMap[i - 1] = newMap;//TODO: Split input and outputs for final stride updates
			types.push_back(DType::String);
			type = DType::String;
		}
		else
			throw invalidCSVFormatException();

		string dir = result[i].substr(3, 3);
		if (strcmp(dir.c_str(), "INP") == 0) {
			//TODO: handle STR/classes
			inOut.push_back(DType::Feature);
			dataset.featureLabels.push_back(result[i].substr(6, string::npos));
		}
		else if (strcmp(dir.c_str(), "OUT") == 0) {
			dataset.outputLabels.push_back(result[i].substr(6, string::npos));

			if (type == DType::Float) {
				//Regression outputs
				inOut.push_back(DType::Regression);
			}
			else {
				//Classification outputs
				inOut.push_back(DType::Classification);
			}
			//TODO: check for other types
		}
		else if (strcmp(dir.c_str(), "ERR") == 0) {
			types[i - 1] = DType::Error;
			errors++;
		}
		else
			throw invalidCSVFormatException();
	}

	int totalLines = 0;
	try {
		totalLines = stoi(result[0]);//First value is number of remaining rows
	}
	catch (exception e) {
		throw invalidCSVFormatException();
	}

	//Set up buffers
	dataset.data = vector<cuDatum>(totalLines);
	int bufferSize = 10000;
	vector<string> lines(bufferSize);//Read buffers

	//Launch parsing threads
	int index = 0;
	queue<thread> parseThreads;

	//Launch all read threads, make a queue of parses
	while (!file.eof()) {
		readCSVThread(lines, file);

		for (int i = 0; i < bufferSize && i < totalLines; i++) {
			parseThreads.push(thread(&parseCSVThread, ref(dataset.data[index++]), lines[i], ref(classMap),
				types, inOut));
		}

		//Empty the parse queue
		while (!parseThreads.empty()) {
			parseThreads.front().join();
			parseThreads.pop();
		}
	}

	//Define total classifications in dictionary
	for (auto iter = classMap.begin(); iter != classMap.end(); ++iter) {
		//TODO: handle input classifications

		dataset.classSizes.push_back((uint)iter->second.size());
	}

	dataset.shouldNormalize = true;
	dataset.normalize();
}

///////////Gererate function

//Recursively generate the data for multidimensional functions
static void sampleFunctions(sharedVector<cuDatum>& dataset, cuDatum& sample, vector<FunctionType>& func,
	float answer, bool randomInputs, int dim, int sampleWidth, float outputNoise,
	uniform_real_distribution<float>& distribution, normal_distribution<float>& distribution2) {
	for (int index = 0; index < sampleWidth; index++) {
		float nextAnswer = answer;

		if (randomInputs) {
			sample.features[dim] = distribution(generator);
		}
		else {
			sample.features[dim] = 2.0f * ((float)index + 1.0f) / ((float)sampleWidth + 1.0f) - 1.0f;
		}
		applyFunction(nextAnswer, sample.features[dim], func[dim]);

		if (dim > 0) {
			sampleFunctions(dataset, sample, func, nextAnswer, randomInputs, dim - 1, sampleWidth, outputNoise, distribution, distribution2);
		}
		else {
			if (outputNoise > 0.0f)
				sample.regressions.front() = nextAnswer + distribution2(generator);
			else
				sample.regressions.front() = nextAnswer;

			dataset.push_back(sample);
		}
	}
}

void generateFunction(cuDataset& dataset, std::vector<FunctionType> funcs, bool randomInputs, float outputNoise, uint samplesPerDimension) {
	if (funcs.size() == 0)
		throw noFunctionsSelectedException();

	if (outputNoise < 0.0f)
		throw negativeNoiseParameterException();

	dataset.clear();
	
	for (int i = 0; i < funcs.size(); i++) {
		dataset.featureLabels.push_back("Axis " + to_string(i));
		dataset.minVals.push_back(-1.0f);
		dataset.maxVals.push_back(1.0f);
	}
	dataset.outputLabels.push_back("Output");

	//Random initialization for potential use
	uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	float temp = outputNoise;
	if (temp <= 0.0f) temp = 1.0f;//distribution decalaration fails if input is 0
	normal_distribution<float> distribution2(0.0, temp / 100.0f);

	uint setSize = (int)pow(samplesPerDimension, (int)funcs.size());

	dataset.data.reserve(setSize);

	cuDatum sample1;
	sample1.features.resize(funcs.size());
	sample1.regressions.resize(1);
	sampleFunctions(dataset.data, sample1, funcs, 1.0f, randomInputs, (int)funcs.size() - 1, samplesPerDimension, outputNoise, distribution, distribution2);

	dataset.shouldNormalize = false;
}

//////////Time Series

inline float seriesFunction(float input, TimeSeriesType type) {
	switch (type)
	{
	case TimeSeriesType::Sine:
		return sin(input * 2.0f * (float)M_PI);
		break;
	case TimeSeriesType::Square:
		if (input < 0.5f) {
			return 1.0f;
		}
		else {
			return -1.0f;
		}
		break;
	case TimeSeriesType::Sawtooth:
		if (input < 0.5f) {
			return -1 + 4 * input;
		}
		else {
			return 1 - 4 * (input - 0.5f);
		}
		break;
	default:
		return 0.0f;
		break;
	}
}

void generateTimeSeries(cuDataset& dataset, std::vector<SeriesSettings> settings, uint samples, uint inputDelay) {
	if (settings.size() == 0)
		throw noFunctionsSelectedException();

	dataset.clear();

	for (int i = 0; i < settings.size(); i++) {
		dataset.featureLabels.push_back("Axis " + to_string(i));
		dataset.minVals.push_back(-1.0f);
		dataset.maxVals.push_back(1.0f);
	}
	dataset.outputLabels.push_back("Output");

	dataset.data.reserve(samples);

	vector<float> inputSamples(inputDelay, 0.0f);
	cuDatum sample1;
	sample1.features.resize(1);
	sample1.regressions.resize(1);
	for (uint i = 0; i < samples + inputDelay; i++) {
		float sum = 0.0f;
		for (uint j = 0; j < settings.size(); j++) {
			SeriesSettings thisSettings = settings[j];
			sum += seriesFunction((float)((i + thisSettings.phaseSamples) % thisSettings.cycleSamples) / thisSettings.cycleSamples, thisSettings.type);
		}
		sample1.regressions[0] = sum;
		sample1.features[0] = inputSamples[i % inputDelay];
		inputSamples[i % inputDelay] = sum;

		if(i >= inputDelay)
			dataset.data.push_back(sample1);
	}

	dataset.shouldNormalize = false;
}

//////////MNIST

//MNIST endian reversal
inline int ReverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void readMNIST(cuDataset& dataset, std::string dataFile, std::string labelFile) {
	ifstream MNISTData(dataFile, ios::binary);
	ifstream MNISTLabel(labelFile, ios::binary);
	if (!MNISTLabel.is_open() || !MNISTData.is_open())
		throw fileNotFoundException();

	dataset.clear();

	int magic_number = 0;
	int number_of_images = 0;
	unsigned int n_rows = 0;
	unsigned int n_cols = 0;
	int numInputs = 0;

	MNISTData.read((char*)&magic_number, sizeof(magic_number));
	MNISTLabel.read((char*)&magic_number, sizeof(magic_number));
	MNISTData.read((char*)&number_of_images, sizeof(number_of_images));
	MNISTLabel.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = ReverseInt(number_of_images);
	MNISTData.read((char*)&n_rows, sizeof(n_rows));
	n_rows = ReverseInt(n_rows);
	MNISTData.read((char*)&n_cols, sizeof(n_cols));
	n_cols = ReverseInt(n_cols);
	numInputs = n_rows * n_cols;

	dataset.featureLabels.push_back("Input Image");
	dataset.outputLabels.push_back("Output Digit Label");
	dataset.classSizes.push_back(10);

	dataset.data.resize(number_of_images);
	for (int i = 0; i < number_of_images; i++) {
		dataset.data[i].features.resize(numInputs);
	}

	vector<unsigned char> tempImage(numInputs);
	unsigned char tempLabel;
	for (int i = 0; i < number_of_images; ++i) {
		MNISTData.read((char*)tempImage.data(), numInputs * sizeof(char));
		copy(tempImage.begin(), tempImage.end(), dataset.data[i].features.begin());

		MNISTLabel.read((char*)&tempLabel, sizeof(char));
		dataset.data[i].classes.push_back((uint)tempLabel);
	}

	for (int i = 0; i < numInputs; i++) {
		dataset.minVals.push_back(0.0f);
		dataset.maxVals.push_back(255.0f);
	}
	dataset.shouldNormalize = false;
}