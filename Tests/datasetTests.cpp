#include "../LinearBackprop/dataset/dataset.h"
#include "../LinearBackprop/dataset/datasetFactories.h"
#include "CppUnitTest.h"
#include <vector>
#include <random>
#include <time.h>

using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace DatasetTest {
	TEST_CLASS(DirectUsage) {
		default_random_engine generator = default_random_engine((unsigned int)time(NULL));
		uniform_int<int> intDistribution = uniform_int_distribution<int>(1, 16);
		uniform_real<float> floatDistribution;

		vector<vector<float> > regressionAnswers;
		vector<vector<uint> > classAnswers;
		vector<vector<float> > features;

		vector<uint> classSizes;

		int numRegression = 5;
		int numClasses = 10;
		int numFeatures = 7;
		int numData = 15;

		uniform_int_distribution<int> classDistribution = uniform_int_distribution<int>(0, numClasses - 1);
		uniform_int_distribution<int> regDistribution = uniform_int_distribution<int>(0, numRegression - 1);
		uniform_int_distribution<int> featureDistribution = uniform_int_distribution<int>(0, numFeatures - 1);
		uniform_int_distribution<int> dataDistribution = uniform_int_distribution<int>(0, numData - 1);
		uniform_int_distribution<int> boolDistribution = uniform_int_distribution<int>(0, 1);

		cuDataset testDataset;
		cuBatch testBatch;

		TEST_METHOD_INITIALIZE(init) {
			//Initialize metadata
			for (int i = 0; i < numClasses; i++) {
				uint value = intDistribution(generator);
				classSizes.push_back(value);
				testDataset.classSizes.push_back(value);
			}

			//Intialize dataset
			for (int datum = 0; datum < numData; datum++) {
				cuDatum thisDatum;
				vector<float> thisRegression;
				for (int i = 0; i < numRegression; i++) {
					float value = floatDistribution(generator);
					thisRegression.push_back(value);
					thisDatum.regressions.push_back(value);
				}
				regressionAnswers.push_back(thisRegression);

				vector<uint> thisClasses;
				for (int i = 0; i < numClasses; i++) {
					uint value = intDistribution(generator);
					thisClasses.push_back(value);
					thisDatum.classes.push_back(value);
				}
				classAnswers.push_back(thisClasses);

				vector<float> thisFeatures;
				for (int i = 0; i < numFeatures; i++) {
					float value = floatDistribution(generator);
					thisFeatures.push_back(value);
					thisDatum.features.push_back(value);
				}
				features.push_back(thisFeatures);

				testDataset.data.push_back(thisDatum);
			}

			testDataset.nextBatch(testBatch);
		}

		//Test the test setup and initialization functions
		TEST_METHOD(DatasetInitCheck) {
			for (int i = 0; i < numClasses; i++) {
				Assert::AreEqual(classSizes[i], (uint)testDataset.classSizes[i]);
			}

			//Intialize dataset
			for (int datum = 0; datum < numData; datum++) {
				for (int i = 0; i < numRegression; i++) {
					Assert::AreEqual(regressionAnswers[datum][i], (float)testDataset.data[datum].regressions[i]);
				}

				for (int i = 0; i < numClasses; i++) {
					Assert::AreEqual(classAnswers[datum][i], (uint)testDataset.data[datum].classes[i]);
				}

				for (int i = 0; i < numFeatures; i++) {
					Assert::AreEqual(features[datum][i], (float)testDataset.data[datum].features[i]);
				}
			}
		}

		///Device
		//Test boundary + random particular class output

		//Test boundary + random particular regression ouptut

		//Test boundary + random particular feature input

		//Check settings stay set after render

		//Test input dataset interface

		//Test output dataset interface

		///Host
		//Test boundary + random particular class output
		TEST_METHOD(ClassAccess) {
			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);
			uint thisData = dataDistribution(generator);
			uint thisClass = classDistribution(generator);

			Assert::AreEqual(classAnswers[thisData][(uint)0], renderedSet.data[thisData].classes[(uint)0]);
			Assert::AreEqual(classAnswers[thisData][(uint)numClasses - 1], renderedSet.data[thisData].classes[(uint)numClasses - 1]);
			Assert::AreEqual(classAnswers[thisData][thisClass], renderedSet.data[thisData].classes[thisClass]);
		}

		//Can't access class as regression
		TEST_METHOD(RegressionClassAccess) {
			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);
			uint thisRegression = regDistribution(generator);

			try {
				int regClass = renderedSet.getClassGroup(thisRegression);
				Assert::Fail();
			}
			catch (classRequestAsRegression) {}

			try {
				int regClass = renderedSet.getClassIndex(thisRegression);
				Assert::Fail();
			}
			catch (classRequestAsRegression) {}
		}

		TEST_METHOD(InvalidOutputAccess) {
			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);
			uint thisOutput = renderedSet.outputs;

			try {
				int noClass = renderedSet.getClassGroup(thisOutput);
				Assert::Fail();
			}
			catch (invalidOutputID) {}

			try {
				int noClass = renderedSet.getClassIndex(thisOutput);
				Assert::Fail();
			}
			catch (invalidOutputID) {}

			try {
				int noClass = renderedSet.getOutputGroup(thisOutput);
				Assert::Fail();
			}
			catch (invalidOutputID) {}

			try {
				float noClass = renderedSet.getOutput(0, thisOutput);
				Assert::Fail();
			}
			catch (invalidOutputID) {}
		}

		//Test boundary + random particular regression ouptut
		TEST_METHOD(RegressionAccess) {
			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);
			uint thisData = dataDistribution(generator);
			uint thisReg = regDistribution(generator);

			Assert::AreEqual(regressionAnswers[thisData][(uint)0], renderedSet.data[thisData].regressions[(uint)0]);
			Assert::AreEqual(regressionAnswers[thisData][(uint)numRegression - 1], renderedSet.data[thisData].regressions[(uint)numRegression - 1]);
			Assert::AreEqual(regressionAnswers[thisData][thisReg], renderedSet.data[thisData].regressions[thisReg]);
		}

		//Test boundary + random particular feature input
		TEST_METHOD(FeatureAccess) {
			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);
			uint thisData = dataDistribution(generator);
			uint thisFeature = regDistribution(generator);

			Assert::AreEqual(features[thisData][(uint)0], renderedSet.data[thisData].features[(uint)0]);
			Assert::AreEqual(features[thisData][(uint)numFeatures - 1], renderedSet.data[thisData].features[(uint)numFeatures - 1]);
			Assert::AreEqual(features[thisData][thisFeature], renderedSet.data[thisData].features[thisFeature]);
		}

		//Check settings stay set after render
		TEST_METHOD(SettingsRender) {
			testDataset.core.testSet = boolDistribution(generator);
			testDataset.core.batchSize = intDistribution(generator);

			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);

			Assert::AreEqual(testDataset.core.testSet, renderedSet.testSet);
			Assert::AreEqual(testDataset.core.batchSize, renderedSet.batchSize);
		}

		//Test input dataset interface
		TEST_METHOD(InputAccess) {
			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);
			uint thisData = dataDistribution(generator);
			renderedSet.batch.selections[0] = thisData;

			for (uint i = 0; i < renderedSet.inputs; i++) {
				Assert::AreEqual(features[thisData][i], renderedSet.getInput(0, i));
			}
		}

		//Test output dataset interface
		TEST_METHOD(OutputAccess) {
			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);
			uint thisData = dataDistribution(generator);
			renderedSet.batch.selections[0] = thisData;

			uint output = 0;
			for (uint i = 0; i < (uint)numRegression; i++) {
				Assert::AreEqual(regressionAnswers[thisData][i], 
					renderedSet.getOutput(0, output++));
			}
			for (uint i = 0; i < (uint)classSizes.size(); i++) {
				for (uint j = 0; j < classSizes[i]; j++) {
					Assert::AreEqual(j == classAnswers[thisData][i] ? 1.0f : 0.0f,
						renderedSet.getOutput(0, output++));
				}
			}
		}

		///MetaData
		//Test derived variables with empty dataset render
		TEST_METHOD(DerivedSettingsEmptySet) {
			cuDataset emptySet;
			testDataset.nextBatch(testBatch, false);
			Dataset renderedSet = emptySet.pushBatchAndRender(false, testBatch);

			Assert::AreEqual((uint)0, renderedSet.outputGroups);
			Assert::AreEqual((uint)0, renderedSet.outputs);
			Assert::AreEqual((uint)0, renderedSet.inputs);
			Assert::AreEqual(true, renderedSet.isEmpty());
		}

		//Test derived variables
		TEST_METHOD(DerivedSettings) {
			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);

			Assert::AreEqual((uint)(numRegression + numClasses), renderedSet.outputGroups);
			Assert::AreEqual((uint)numFeatures, renderedSet.inputs);
			Assert::AreEqual(false, renderedSet.isEmpty());

			uint outputs = numRegression;
			for (uint i = 0; i < classSizes.size(); i++)
				outputs += classSizes[i];
			Assert::AreEqual(outputs, renderedSet.outputs);
		}

		TEST_METHOD(EmptySet) {
			cuDataset emptySet;

			try {
				Dataset test = emptySet.render(false);
				Assert::Fail();
			}
			catch (noBatchException) { }
		}

		//Test getData with batch call
		TEST_METHOD(GetDatum) {
			testDataset.core.testSet = true;
			int selection = testBatch.selections[0];
			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);

			Datum testData = renderedSet[0];

			uint thisClass = classDistribution(generator);
			uint thisReg = regDistribution(generator);
			uint thisFeature = regDistribution(generator);

			Assert::AreEqual(features[selection][thisFeature], testData.features[thisFeature]);
			Assert::AreEqual(regressionAnswers[selection][thisReg], testData.regressions[thisReg]);
			Assert::AreEqual(classAnswers[selection][thisClass], testData.classes[thisClass]);
		}

		//Datum out of bounds
		TEST_METHOD(DatumOutOfBounds) {
			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);

			try {
				Datum testData = renderedSet[1];
				Assert::Fail();
			}
			catch (outOfBounds) {}
		}

		//Input out of bounds
		TEST_METHOD(InputOutOfBounds) {
			Dataset renderedSet = testDataset.pushBatchAndRender(false, testBatch);

			try {
				float testData = renderedSet.getInput(0, numFeatures);
				Assert::Fail();
			}
			catch (outOfBounds) {}
		}

		//Test dataset copy
		TEST_METHOD(DatasetCopy) {
			cuDataset copySet = testDataset;

			//Test shared vectors
			for (int i = 0; i < copySet.data.size(); i++) {
				for(int j = 0; j < testDataset.data[i].regressions.size(); j++)
					Assert::AreEqual((float)testDataset.data[i].regressions[j], (float)copySet.data[i].regressions[j]);
				for (int j = 0; j < testDataset.data[i].features.size(); j++)
					Assert::AreEqual((float)testDataset.data[i].features[j], (float)copySet.data[i].features[j]);
				for (int j = 0; j < testDataset.data[i].classes.size(); j++)
					Assert::AreEqual((int)testDataset.data[i].classes[j], (int)copySet.data[i].classes[j]);
			}
			for (int i = 0; i < copySet.featureLabels.size(); i++) {
				Assert::IsTrue(copySet.featureLabels[i].compare(testDataset.featureLabels[i]));
			}
			for (int i = 0; i < copySet.outputLabels.size(); i++) {
				Assert::IsTrue(copySet.outputLabels[i].compare(testDataset.outputLabels[i]));
			}
			for (int i = 0; i < copySet.classSizes.size(); i++) {
				Assert::AreEqual((uint)testDataset.classSizes[i], (uint)copySet.classSizes[i]);
			}

			//Test settings in core
			Assert::AreEqual(testDataset.core.batchSize, copySet.core.batchSize);
			Assert::AreEqual(testDataset.core.testSet, copySet.core.testSet);
		}

		//Test split inputs can only be between 0 and 1
		TEST_METHOD(SplitBoundaries) {
			cuDataset other;

			try {
				testDataset.split(other, -1.0f, true);
				Assert::Fail();
			}
			catch (invalidSplitParameter) {}

			try {
				testDataset.split(other, 10000.0f, true);
				Assert::Fail();
			}
			catch (invalidSplitParameter) {}

			try {
				testDataset.split(other, 0.0f, true);
				Assert::Fail();
			}
			catch (invalidSplitParameter) {}

			try {
				testDataset.split(other, 1.0f, true);
				Assert::Fail();
			}
			catch (invalidSplitParameter) {}
		}

		//Test split carried over all settings
		TEST_METHOD(SplitSettings) {
			cuDataset splitSet;
			testDataset.split(splitSet, 0.5, false);

			for (int i = 0; i < splitSet.featureLabels.size(); i++) {
				Assert::IsTrue(splitSet.featureLabels[i].compare(testDataset.featureLabels[i]));
			}
			for (int i = 0; i < splitSet.outputLabels.size(); i++) {
				Assert::IsTrue(splitSet.outputLabels[i].compare(testDataset.outputLabels[i]));
			}
			for (int i = 0; i < splitSet.classSizes.size(); i++) {
				Assert::AreEqual((uint)testDataset.classSizes[i], (uint)splitSet.classSizes[i]);
			}

			//Test settings in core
			Assert::AreEqual(testDataset.core.batchSize, splitSet.core.batchSize);
			Assert::AreEqual(testDataset.core.testSet, splitSet.core.testSet);
		}

		//Test split accurately split dataset
		TEST_METHOD(SplitAccuracy) {
			cuDataset verificationSet = testDataset;
			cuDataset splitSet;
			testDataset.split(splitSet, 0.5, false);
			testDataset.revertNormalization();
			splitSet.revertNormalization();
			uint splitSetSize = (uint)testDataset.data.size();

			Assert::AreEqual((size_t)(verificationSet.data.size() * 0.5f), testDataset.data.size());
			Assert::AreEqual(verificationSet.data.size() - (size_t)(verificationSet.data.size() * 0.5f), splitSet.data.size());

			for (int i = 0; i < testDataset.data.size(); i++) {
				for (int j = 0; j < testDataset.data[i].regressions.size(); j++)
					Assert::AreEqual((float)testDataset.data[i].regressions[j], (float)verificationSet.data[i].regressions[j], 0.0001f);
				for (int j = 0; j < testDataset.data[i].features.size(); j++)
					Assert::AreEqual((float)testDataset.data[i].features[j], (float)verificationSet.data[i].features[j], 0.0001f);
				for (int j = 0; j < testDataset.data[i].classes.size(); j++)
					Assert::AreEqual((int)testDataset.data[i].classes[j], (int)verificationSet.data[i].classes[j]);
			}

			for (int i = 0; i < splitSet.data.size(); i++) {
				for (int j = 0; j < splitSet.data[i].regressions.size(); j++)
					Assert::AreEqual((float)splitSet.data[i].regressions[j], (float)verificationSet.data[splitSetSize + i].regressions[j], 0.0001f);
				for (int j = 0; j < splitSet.data[i].features.size(); j++)
					Assert::AreEqual((float)splitSet.data[i].features[j], (float)verificationSet.data[splitSetSize + i].features[j], 0.0001f);
				for (int j = 0; j < splitSet.data[i].classes.size(); j++)
					Assert::AreEqual((int)splitSet.data[i].classes[j], (int)verificationSet.data[splitSetSize + i].classes[j]);
			}
		}

		//Test random split at least gives the right sizes
		TEST_METHOD(SplitRandom) {
			cuDataset verificationSet = testDataset;
			cuDataset splitSet;
			testDataset.split(splitSet, 0.5, true);
			
			Assert::AreEqual((size_t)(verificationSet.data.size() * 0.5f), testDataset.data.size());
			Assert::AreEqual(verificationSet.data.size() - (size_t)(verificationSet.data.size() * 0.5f), splitSet.data.size());
		}

		//Split and merge should reproduce original
		TEST_METHOD(SplitAndMerge) {
			cuDataset verificationSet = testDataset;
			cuDataset splitSet;
			testDataset.split(splitSet, 0.5, false);

			testDataset.merge(splitSet);
			testDataset.revertNormalization();

			Assert::AreEqual(verificationSet.data.size(), testDataset.data.size());

			for (int i = 0; i < testDataset.data.size(); i++) {
				for (int j = 0; j < testDataset.data[i].regressions.size(); j++)
					Assert::AreEqual((float)testDataset.data[i].regressions[j], (float)verificationSet.data[i].regressions[j], 0.0001f);
				for (int j = 0; j < testDataset.data[i].features.size(); j++)
					Assert::AreEqual((float)testDataset.data[i].features[j], (float)verificationSet.data[i].features[j], 0.0001f);
				for (int j = 0; j < testDataset.data[i].classes.size(); j++)
					Assert::AreEqual((int)testDataset.data[i].classes[j], (int)verificationSet.data[i].classes[j]);
			}
		}

		//Test window invalid inputs
		TEST_METHOD(WindowBoundaries) {
			cuDataset other;

			try {
				testDataset.getWindow(other, -1.0f, 0.5f);
				Assert::Fail();
			}
			catch (invalidWindowParameter) {}

			try {
				testDataset.getWindow(other, 10000.0f, 0.5f);
				Assert::Fail();
			}
			catch (invalidWindowParameter) {}

			try {
				testDataset.getWindow(other, 0.5f, -1.0f);
				Assert::Fail();
			}
			catch (invalidWindowParameter) {}

			try {
				testDataset.getWindow(other, 0.5f, 10000.0f);
				Assert::Fail();
			}
			catch (invalidWindowParameter) {}

			try {
				testDataset.getWindow(other, 0.0f, 0.5f);
				Assert::Fail();
			}
			catch (invalidWindowParameter) {}

			try {
				testDataset.getWindow(other, 0.5f, 1.0f);
				Assert::Fail();
			}
			catch (invalidWindowParameter) {}

			try {
				testDataset.getWindow(other, 0.75f, 0.75f);
				Assert::Fail();
			}
			catch (invalidWindowParameter) {}
		}

		//Test window carried over all settings
		TEST_METHOD(WindowSettings) {
			cuDataset splitSet;
			testDataset.getWindow(splitSet, 0.5, 0.0f);

			for (int i = 0; i < splitSet.featureLabels.size(); i++) {
				Assert::IsTrue(splitSet.featureLabels[i].compare(testDataset.featureLabels[i]));
			}
			for (int i = 0; i < splitSet.outputLabels.size(); i++) {
				Assert::IsTrue(splitSet.outputLabels[i].compare(testDataset.outputLabels[i]));
			}
			for (int i = 0; i < splitSet.classSizes.size(); i++) {
				Assert::AreEqual((uint)testDataset.classSizes[i], (uint)splitSet.classSizes[i]);
			}

			//Test settings in core
			Assert::AreEqual(testDataset.core.batchSize, splitSet.core.batchSize);
			Assert::AreEqual(testDataset.core.testSet, splitSet.core.testSet);
		}

		//Test window accurately copied dataset
		TEST_METHOD(WindowAccuracy) {
			cuDataset splitSet;
			float offsetPercent = 0.1f;
			float windowPercent = 0.5f;
			testDataset.getWindow(splitSet, windowPercent, offsetPercent);
			testDataset.revertNormalization();
			splitSet.revertNormalization();
			uint offset = (uint)(offsetPercent * testDataset.data.size());

			Assert::AreEqual((size_t)(testDataset.data.size() * windowPercent), splitSet.data.size());

			for (int i = 0; i < splitSet.data.size(); i++) {
				for (int j = 0; j < splitSet.data[i].regressions.size(); j++)
					Assert::AreEqual((float)splitSet.data[i].regressions[j], (float)testDataset.data[offset + i].regressions[j], 0.0001f);
				for (int j = 0; j < splitSet.data[i].features.size(); j++)
					Assert::AreEqual((float)splitSet.data[i].features[j], (float)testDataset.data[offset + i].features[j], 0.0001f);
				for (int j = 0; j < splitSet.data[i].classes.size(); j++)
					Assert::AreEqual((int)splitSet.data[i].classes[j], (int)testDataset.data[offset + i].classes[j]);
			}
		}

		//Test get inputs
		TEST_METHOD(GetInputs) {
			uint inputID = uniform_int_distribution<int>(0, numFeatures - 1)(generator);
			vector<float> inputs = testDataset.getInputs(inputID);

			Assert::AreEqual(numData, (int)inputs.size());

			for (uint i = 0; i < features.size(); i++) {
				Assert::AreEqual(features[i][inputID], inputs[i]);
			}
		}

		//Test get outputs
		TEST_METHOD(GetOutputs) {
			uint outputID = uniform_int_distribution<int>(0, numRegression - 1)(generator);
			vector<float> outputs = testDataset.getOutputs(outputID);

			Assert::AreEqual(numData, (int)outputs.size());

			for (uint i = 0; i < features.size(); i++) {
				Assert::AreEqual(regressionAnswers[i][outputID], outputs[i]);
			}
		}

		//Test get for classOutputs
		TEST_METHOD(GetClassOutputs) {
			uint outputID = uniform_int_distribution<int>(0, numClasses - 1)(generator);
			vector<float> outputs = testDataset.getOutputs(outputID + numRegression);

			Assert::AreEqual(numData, (int)outputs.size());

			for (uint i = 0; i < classAnswers.size(); i++) {
				Assert::AreEqual(classAnswers[i][outputID], (uint)outputs[i]);
			}
		}

		//Test out of bounds get exceptions
		TEST_METHOD(GetExceptions) {
			try {
				vector<float> outputs = testDataset.getOutputs(numRegression + numClasses);
				Assert::Fail();
			}
			catch (invalidOutputID) {}

			try {
				vector<float> inputs = testDataset.getInputs(numFeatures);
				Assert::Fail();
			}
			catch (invalidInputID) {}
		}

		//Test normalization
		TEST_METHOD(SetNormalization) {
			testDataset.normalize();

			//For each input set
			for (uint i = 0; i < testDataset.inputs(); i++) {
				//Get mean
				float mean = 0.0f;
				for (uint j = 0; j < testDataset.data.size(); j++) {
					mean += testDataset.data[j].features[i];
				}
				mean /= testDataset.data.size();

				//Mean should be 0 after
				Assert::AreEqual(0.0f, mean, 0.001f);

				//Get stdDev
				float stdDev = 0.0f;
				for (uint j = 0; j < testDataset.data.size(); j++) {
					stdDev += pow(testDataset.data[j].features[i] - mean, 2.0f);
				}
				stdDev = sqrt(stdDev / testDataset.data.size());

				//Std Dev should be 1
				Assert::AreEqual(1.0f, stdDev, 0.001f);
			}

			//For each output set
			for (uint i = 0; i < (uint)testDataset.data[0].regressions.size(); i++) {
				//Get mean
				float mean = 0.0f;
				for (uint j = 0; j < testDataset.data.size(); j++) {
					mean += testDataset.data[j].regressions[i];
				}
				mean /= testDataset.data.size();

				//Mean should be 0 after
				Assert::AreEqual(0.0f, mean, 0.001f);

				//Get stdDev
				float stdDev = 0.0f;
				for (uint j = 0; j < testDataset.data.size(); j++) {
					stdDev += pow(testDataset.data[j].regressions[i] - mean, 2.0f);
				}
				stdDev = sqrt(stdDev / testDataset.data.size());

				//Std Dev should be 1
				Assert::AreEqual(1.0f, stdDev, 0.001f);
			}
		}

		//Test normalization reset
		TEST_METHOD(SetRevertNormalization) {
			//Normalize and revert
			cuDataset verificationSet = testDataset;
			testDataset.normalize();
			testDataset.revertNormalization();

			//Set should be equal to original (with FLT loss)
			for (int i = 0; i < testDataset.data.size(); i++) {
				for (int j = 0; j < testDataset.data[i].regressions.size(); j++)
					Assert::AreEqual(verificationSet.data[i].regressions[j], testDataset.data[i].regressions[j], 0.0001f);
				for (int j = 0; j < testDataset.data[i].features.size(); j++)
					Assert::AreEqual(verificationSet.data[i].features[j], testDataset.data[i].features[j], 0.0001f);
			}
		}

		//Test that getting inputs and outputs has revert applied
		TEST_METHOD(GetNormalizedData) {
			cuDataset verificationSet = testDataset;
			testDataset.normalize();

			//GetInputs equal to inputs
			for (int i = 0; i < (int)testDataset.inputs(); i++) {
				vector<float> testVec = testDataset.getInputs(i);
				for (int j = 0; j < testDataset.data.size(); j++) {
					Assert::AreEqual(verificationSet.data[j].features[i], testVec[j], 0.0001f);
				}
			}

			//GetOutputs equal to outputs
			for (int i = 0; i < (int)testDataset.data[0].regressions.size(); i++) {
				vector<float> testVec = testDataset.getOutputs(i);
				for (int j = 0; j < testDataset.data.size(); j++) {
					Assert::AreEqual(verificationSet.data[j].regressions[i], testVec[j], 0.0001f);
				}
			}
		}
	};

	TEST_CLASS(Batch) {
		default_random_engine generator = default_random_engine((unsigned int)time(NULL));
		uniform_int<int> intDistribution;
		uniform_real<float> floatDistribution;

		vector<vector<float> > regressionAnswers;
		vector<vector<uint> > classAnswers;
		vector<vector<float> > features;

		vector<uint> classSizes;

		int numRegression = 5;
		int numClasses = 10;
		int numFeatures = 7;
		int numData = 15;

		uniform_int_distribution<int> classDistribution = uniform_int_distribution<int>(0, numClasses - 1);
		uniform_int_distribution<int> regDistribution = uniform_int_distribution<int>(0, numRegression - 1);
		uniform_int_distribution<int> featureDistribution = uniform_int_distribution<int>(0, numFeatures - 1);
		uniform_int_distribution<int> dataDistribution = uniform_int_distribution<int>(0, numData - 1);
		uniform_int_distribution<int> boolDistribution = uniform_int_distribution<int>(0, 1);

		cuDataset testDataset;
		cuBatch testBatch;

		TEST_METHOD_INITIALIZE(init) {
			//Initialize metadata
			for (int i = 0; i < numClasses; i++) {
				uint value = intDistribution(generator);
				classSizes.push_back(value);
				testDataset.classSizes.push_back(value);
			}

			//Intialize dataset
			for (int datum = 0; datum < numData; datum++) {
				cuDatum thisDatum;
				vector<float> thisRegression;
				for (int i = 0; i < numRegression; i++) {
					float value = floatDistribution(generator);
					thisRegression.push_back(value);
					thisDatum.regressions.push_back(value);
				}
				regressionAnswers.push_back(thisRegression);

				vector<uint> thisClasses;
				for (int i = 0; i < numClasses; i++) {
					uint value = intDistribution(generator);
					thisClasses.push_back(value);
					thisDatum.classes.push_back(value);
				}
				classAnswers.push_back(thisClasses);

				vector<float> thisFeatures;
				for (int i = 0; i < numFeatures; i++) {
					float value = floatDistribution(generator);
					thisFeatures.push_back(value);
					thisDatum.features.push_back(value);
				}
				features.push_back(thisFeatures);

				testDataset.data.push_back(thisDatum);
			}

			testDataset.nextBatch(testBatch, false);
		}

		///Standard batch
		//Test allocations
		TEST_METHOD(Allocations) {
			int batchTests = 8;
			testDataset.core.batchSize = batchTests;
			testDataset.nextBatch(testBatch, false);

			//Test allocations
			Assert::AreEqual((size_t)batchTests, testBatch.selections.size());
			Assert::AreEqual((size_t)batchTests, testBatch.responses.size());
				Assert::AreEqual((size_t)testDataset.outputs(), testBatch.responses.front().size());
				Assert::AreEqual((size_t)testDataset.outputs(), testBatch.responses.back().size());
			Assert::AreEqual((size_t)batchTests, testBatch.classResponses.size());
				Assert::AreEqual((size_t)numClasses, testBatch.classResponses.front().size());
				Assert::AreEqual((size_t)numClasses, testBatch.classResponses.back().size());
			Assert::AreEqual((size_t)numClasses + numRegression, testBatch.statistics.size());
		}

		//Train generate set check inside boundary (can't check random)
		TEST_METHOD(RandomTrainSets) {
			uint batchTests = 8;
			testDataset.core.batchSize = batchTests;
			testDataset.core.testSet = false;
			testDataset.nextBatch(testBatch);
			
			for (uint i = 0; i < testBatch.selections.size(); i++) {
				Assert::IsTrue(testBatch.selections[i] >= 0);
				Assert::IsTrue(testBatch.selections[i] < testDataset.data.size());
			}
		}

		//Test generate set standard counting
		TEST_METHOD(SequentialTestSets) {
			uint batchTests = 8;
			testDataset.core.batchSize = batchTests;
			testDataset.core.testSet = true;
			testDataset.nextBatch(testBatch);

			for (uint i = 0; i < testBatch.selections.size(); i++) {
				Assert::AreEqual(i, (uint)testBatch.selections[i]);
			}
		}

		//Test generate set end of dataset
		TEST_METHOD(DatasetEndTest) {
			uint batchTests = 8;
			testDataset.core.batchSize = batchTests;
			testDataset.core.testSet = true;
			testDataset.nextBatch(testBatch);

			for (uint batchID = 0; batchID < overDiv((uint)testDataset.data.size(), batchTests); batchID++) {
				for (uint i = 0; i < testBatch.selections.size(); i++) {
					Assert::AreEqual(batchTests * batchID + i, (uint)testBatch.selections[i]);
				}
				testDataset.nextBatch(testBatch, false, &testBatch);
			}
		}

		//Test generate set batch larger than dataset
		TEST_METHOD(OversizedBatch) {
			uint batchTests = 1024;
			testDataset.core.batchSize = batchTests;
			testDataset.core.testSet = true;
			testDataset.nextBatch(testBatch);

			for (uint batchID = 0; batchID < overDiv((uint)testDataset.data.size(), batchTests); batchID++) {
				for (uint i = 0; i < testBatch.selections.size(); i++) {
					Assert::AreEqual(batchTests * batchID + i, (uint)testBatch.selections[i]);
				}
				testDataset.nextBatch(testBatch, &testBatch);
			}
		}

		//Test set should reset after end
		TEST_METHOD(TestBatchReset) {
			uint batchTests = 8;
			testDataset.core.batchSize = batchTests;
			testDataset.core.testSet = true;
			testDataset.nextBatch(testBatch);

			for (int i = 0; i < 2; i++) {
				for (uint batchID = 0; batchID < overDiv((uint)testDataset.data.size(), batchTests); batchID++) {
					for (uint j = 0; j < testBatch.selections.size(); j++) {
						Assert::AreEqual(batchTests * batchID + j, (uint)testBatch.selections[j]);
					}
					testDataset.nextBatch(testBatch, false, &testBatch);
				}
			}
		}

		//Test set should reset after end
		TEST_METHOD(BatchPremake) {
			uint batchTests = 8;
			testDataset.core.batchSize = batchTests;
			testDataset.core.testSet = true;
			testDataset.nextBatch(testBatch);

			for (int i = 0; i < 2; i++) {
				uint batches = overDiv((uint)testDataset.data.size(), batchTests);
				vector<cuBatch> batchVec(batches);
				testDataset.nextBatch(batchVec[0]);
				for (uint batchID = 1; batchID < batches; batchID++)
					testDataset.nextBatch(batchVec[batchID], false, &batchVec[batchID -1]);

				for (uint batchID = 0; batchID < batches; batchID++) {
					for (uint j = 0; j < batchVec[batchID].selections.size(); j++) {
						Assert::AreEqual(batchTests * batchID + j, (uint)batchVec[batchID].selections[j]);
					}
				}
			}
		}

		///Recurrent
		//Intialization uses stride
		TEST_METHOD(RecurrentBatchInit) {
			uint batchTests = 8;
			testDataset.core.batchSize = batchTests;
			testDataset.nextBatch(testBatch, true);
			uint batches = overDiv((uint)testDataset.data.size(), batchTests);

			for (uint i = 0; i < batchTests; i++) {
				Assert::AreEqual(batches * i, (uint)testBatch.selections[i]);
			}
		}

		//Additional sets increment original
		TEST_METHOD(RecurrentBatchIncrement) {
			uint batchTests = 8;
			testDataset.core.batchSize = batchTests;
			testDataset.nextBatch(testBatch, true);
			uint batches = overDiv((uint)testDataset.data.size(), batchTests);

			for (uint batchID = 0; batchID < overDiv((uint)testDataset.data.size(), batchTests); batchID++) {
				for (uint i = 0; i < batchTests; i++) {
					Assert::AreEqual(batches * i + batchID, (uint)testBatch.selections[i]);
				}
				testDataset.nextBatch(testBatch, true, &testBatch);
			}
		}

		//Restart after dataset ends
		TEST_METHOD(RecurrentBatchReset) {
			uint batchTests = 8;
			testDataset.core.batchSize = batchTests;
			testDataset.nextBatch(testBatch, true);
			uint batches = overDiv((uint)testDataset.data.size(), batchTests);//Reserve one extra to force reset

			for (int repeat = 0; repeat < 2; repeat++) {
				for (uint batchID = 0; batchID < batches; batchID++) {
					for (uint i = 0; i < batchTests; i++) {
						Assert::AreEqual(batches * i + batchID, (uint)testBatch.selections[i]);
					}
					testDataset.nextBatch(testBatch, true, &testBatch);
				}
			}
		}
	};

	//Test out of bounds class entry (factory)

	TEST_CLASS(CSVFactory) {
		//Test file not found exception
		TEST_METHOD(FileExceptions) {
			try {
				cuDataset testSet;
				readCSV(testSet, "invalid");
				Assert::Fail();
			}
			catch (fileNotFoundException) {}
		}

		//Test bad CSV formats
		TEST_METHOD(CSVExceptions) {
			try {
				cuDataset testSet;
				readCSV(testSet, "D:\\repos\\Dynamic Network\\NetTest\\TestFiles\\irisBadType.csv");
				Assert::Fail();
			}
			catch (invalidCSVFormatException) {}

			try {
				cuDataset testSet;
				readCSV(testSet, "D:\\repos\\Dynamic Network\\NetTest\\TestFiles\\irisBadInOut.csv");
				Assert::Fail();
			}
			catch (invalidCSVFormatException) {}
		}

		//Test expected line numbers and data types (regression)
		TEST_METHOD(RegressionRead) {
			cuDataset testSet;
			readCSV(testSet, "D:\\repos\\Dynamic Network\\NetTest\\TestFiles\\simpleSeriesDataset.csv");

			Assert::AreEqual(99, (int)testSet.data.size());
			Assert::AreEqual(1, (int)testSet.data[0].features.size());
			Assert::AreEqual(0, (int)testSet.data[0].classes.size());
			Assert::AreEqual(1, (int)testSet.data[0].regressions.size());

			Assert::AreEqual(1, (int)testSet.minVals.size());
			Assert::AreEqual(1, (int)testSet.maxVals.size());

			Assert::AreEqual(0, (int)testSet.classSizes.size());
			Assert::IsTrue(testSet.featureLabels[0].compare("Series Data") == 0);
			Assert::IsTrue(testSet.outputLabels[0].compare("Series Forecast") == 0);
		}

		//Test expected line numbers and data types (classes)
		TEST_METHOD(ClassRead) {
			cuDataset testSet;
			readCSV(testSet, "D:\\repos\\Dynamic Network\\NetTest\\TestFiles\\iris.csv");

			Assert::AreEqual(150, (int)testSet.data.size());
			Assert::AreEqual(4, (int)testSet.data[0].features.size());
			Assert::AreEqual(1, (int)testSet.data[0].classes.size());
			Assert::AreEqual(0, (int)testSet.data[0].regressions.size());

			Assert::AreEqual(4, (int)testSet.minVals.size());
			Assert::AreEqual(4, (int)testSet.maxVals.size());

			Assert::AreEqual(1, (int)testSet.classSizes.size());
			Assert::AreEqual(3, (int)testSet.classSizes[0]);
			Assert::IsTrue(testSet.featureLabels[0].compare("Sepal length") == 0);
			Assert::IsTrue(testSet.featureLabels[1].compare("Sepal width") == 0);
			Assert::IsTrue(testSet.featureLabels[2].compare("Petal length") == 0);
			Assert::IsTrue(testSet.featureLabels[3].compare("Petal width") == 0);
			Assert::IsTrue(testSet.outputLabels[0].compare("Class") == 0);
		}

		//Test reordering of columns has the same data
		TEST_METHOD(Reorder) {
			//Do we even want this?
					   
			//cuDataset testSet = readCSV("D:\\repos\\LinearBackprop\\NetTest\\TestFiles\\iris.csv");


		}
	};

	TEST_CLASS(MNISTFactory) {
		//Test file not found exception data and label files
		TEST_METHOD(FileExceptions) {
			try {
				cuDataset testSet;
				readMNIST(testSet, "D:\\repos\\LinearBackprop\\NetTest\\TestFiles\\t10k-images.idx3-ubyte", "invalid");
				Assert::Fail();
			}
			catch (fileNotFoundException) {}

			try {
				cuDataset testSet;
				readMNIST(testSet, "invalid", "D:\\repos\\LinearBackprop\\NetTest\\TestFiles\\t10k-labels.idx1-ubyte");
				Assert::Fail();
			}
			catch (fileNotFoundException) {}
		}

		//Test dataset, featureset and output sizes, as well as names
		TEST_METHOD(MetadataValidity) {
			cuDataset testSet;
			readMNIST(testSet, "D:\\repos\\Dynamic Network\\NetTest\\TestFiles\\t10k-images.idx3-ubyte", "D:\\repos\\Dynamic Network\\NetTest\\TestFiles\\t10k-labels.idx1-ubyte");

			Assert::AreEqual(10000, (int)testSet.data.size());
			Assert::AreEqual(784, (int)testSet.data[0].features.size());
			Assert::AreEqual(1, (int)testSet.data[0].classes.size());
			Assert::AreEqual(0, (int)testSet.data[0].regressions.size());

			Assert::AreEqual(784, (int)testSet.minVals.size());
			Assert::AreEqual(784, (int)testSet.maxVals.size());
			for (int i = 0; i < 784; i++) {
				Assert::AreEqual(0, (int)testSet.minVals[i]);
				Assert::AreEqual(255, (int)testSet.maxVals[i]);
			}

			Assert::AreEqual(1, (int)testSet.classSizes.size());
			Assert::AreEqual(10, (int)testSet.classSizes[0]);
			Assert::IsTrue(testSet.featureLabels[0].compare("Input Image") == 0);
			Assert::IsTrue(testSet.outputLabels[0].compare("Output Digit Label") == 0);
		}
	};

	TEST_CLASS(FunctionFactory) {
		//Test exception on empty vector
		TEST_METHOD(EmptyVector) {
			vector<FunctionType> funcs = {};

			try {
				cuDataset testSet;
				generateFunction(testSet, funcs, false, 0.0, 2);
				Assert::Fail();
			}
			catch (noFunctionsSelectedException) {}
		}

		//Test exception on negative output noise
		TEST_METHOD(NegativeNoise) {
			vector<FunctionType> funcs = { FunctionType::None, FunctionType::Line };

			try {
				cuDataset testSet;
				generateFunction(testSet, funcs, false, -1.0f, 2);
				Assert::Fail();
			}
			catch (negativeNoiseParameterException) {}
		}

		//Test correct usage
		TEST_METHOD(MetadataValidity) {
			vector<FunctionType> funcs = { FunctionType::Line, FunctionType::Quadratic };
			cuDataset testSet;
			uint samples = 10;
			generateFunction(testSet, funcs, false, 0.0, samples);

			Assert::AreEqual((int)pow(samples, (int)funcs.size()), (int)testSet.data.size());
			Assert::AreEqual((int)funcs.size(), (int)testSet.data[0].features.size());
			Assert::AreEqual(0, (int)testSet.data[0].classes.size());
			Assert::AreEqual(1, (int)testSet.data[0].regressions.size());

			Assert::AreEqual((int)funcs.size(), (int)testSet.minVals.size());
			Assert::AreEqual((int)funcs.size(), (int)testSet.maxVals.size());
			for (int i = 0; i < (int)funcs.size(); i++) {
				Assert::AreEqual(-1.0f, testSet.minVals[i], 0.3f);
				Assert::AreEqual(1.0f, testSet.maxVals[i], 0.3f);
			}

			Assert::AreEqual(0, (int)testSet.classSizes.size());
			Assert::IsTrue(testSet.featureLabels[0].compare("Axis 0") == 0);
			Assert::IsTrue(testSet.featureLabels[1].compare("Axis 1") == 0);
			Assert::IsTrue(testSet.outputLabels[0].compare("Output") == 0);
		}

		//Preserve settings after new batch is generated
		TEST_METHOD(SettingsConsistency) {
			vector<FunctionType> funcs = { FunctionType::None, FunctionType::Line };
			cuDataset testSet;
			generateFunction(testSet, funcs, false, 0.0, 2);
			uint testBatchSize = 42;
			bool isTest = true;
			bool recurrent = true;

			testSet.core.batchSize = testBatchSize;
			testSet.core.testSet = isTest;

			generateFunction(testSet, funcs, false, 0.0, 2);

			Assert::AreEqual(testBatchSize, testSet.core.batchSize);
			Assert::AreEqual(isTest, testSet.core.testSet);
		}
	};
}