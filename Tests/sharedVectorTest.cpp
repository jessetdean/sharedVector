#include "CppUnitTest.h"
#include <vector>
#include <random>
#include <time.h>
#include "../LinearBackprop/sharedVector/sharedVector.h"

using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace SharedVectorTest {
	TEST_CLASS(StdVectorConstructors) {
		default_random_engine generator;
		uniform_int<int> intDistribution;
		vector<int> baseVector;
		int testItems = 5;

		TEST_METHOD_INITIALIZE(init) {
			for (int i = 0; i < testItems; i++) {
				baseVector.push_back(intDistribution(generator));
			}
		}

		TEST_METHOD(VectorBasicUsage) {
			sharedVector<int> testVec;
			Assert::AreEqual((int)testVec.size(), 0);

			int value = intDistribution(generator);
			testVec.push_back(value);
			Assert::AreEqual((int)testVec.size(), 1);
			Assert::AreEqual((int)testVec[0], value);
		}

		//Test external vector initialization
		TEST_METHOD(VectorInitialization) {
			sharedVector<int> testVec({ 1, 2, 3, 4 });
			Assert::AreEqual((int)testVec.size(), 4);
			Assert::AreEqual((int)testVec[0], 1);
			Assert::AreEqual((int)testVec[1], 2);
			Assert::AreEqual((int)testVec[2], 3);
			Assert::AreEqual((int)testVec[3], 4);
		}

		//Copy constructor
		TEST_METHOD(StdCopy)
		{
			//Run constructor
			sharedVector<int> testVector(baseVector);
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual((int)baseVector[i], (int)testVector[i]);
			}
		}

		//Copy assignment
		TEST_METHOD(StdCopyAssignment)
		{
			//Run constructor
			sharedVector<int> testVector = baseVector;
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual((int)baseVector[i], (int)testVector[i]);
			}
		}

		//Move constructor
		TEST_METHOD(StdMove)
		{
			//Run constructor
			vector<int> baseCopy = baseVector;
			sharedVector<int> testVector(move(baseVector));
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual((int)baseCopy[i], (int)testVector[i]);
			}
			Assert::AreEqual(0, (int)baseVector.size());
		}

		//Move assignment
		TEST_METHOD(StdMoveAssignment)
		{
			//Run constructor
			vector<int> baseCopy = baseVector;
			sharedVector<int> testVector = move(baseVector);
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual((int)baseCopy[i], (int)testVector[i]);
			}
			Assert::AreEqual(0, (int)baseVector.size());
		}
	};

	TEST_CLASS(DeviceConstructors)
	{
	public:
		default_random_engine generator = default_random_engine((unsigned int)time(NULL));

		uniform_int<int> intDistribution;
		vector<int> baseVectorInt;
		sharedVector<int> testVectorInt;

		uniform_real<float> floatDistribution;
		vector<float> baseVectorFloat;
		sharedVector<float> testVectorFloat;
		int testItems = 5;

		TEST_METHOD_INITIALIZE(init) {
			for (int i = 0; i < testItems; i++) {
				int newVal = intDistribution(generator);
				baseVectorInt.push_back(newVal);
				testVectorInt.push_back(newVal);

				float newValFloat = floatDistribution(generator);
				baseVectorFloat.push_back(newValFloat);
				testVectorFloat.push_back(newValFloat);
			}
		}

		//Test basic render function (device)
		TEST_METHOD(RenderInt)
		{
			//Send to device
			cudaVector<int> deviceTest = testVectorInt;

			//Sync back to host
			testVectorInt.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual(baseVectorInt[i], (int)testVectorInt[i]);
			}
		}

		//Test render and syncs for floats
		TEST_METHOD(RenderFloat)
		{
			//Send to device
			cudaVector<float> deviceTest = testVectorFloat;

			//Sync back to host
			testVectorFloat.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual(baseVectorFloat[i], (float)testVectorFloat[i]);
			}
		}

		//Test quoted size (device)
		TEST_METHOD(TestSizeQuote) {
			sharedVector<float> testVector;

			Assert::AreEqual(sizeof(cudaVector<float>), testVector.byteSize());

			testVector.push_back(0.0f);

			Assert::AreEqual(sizeof(cudaVector<float>), testVector.byteSize());
		}

		//Self copy test (device)
		TEST_METHOD(SelfAssignment)
		{
			//Send to device
			cudaVector<float> deviceTest = testVectorFloat;

			//Self assignment
			testVectorFloat = testVectorFloat;

			//Sync back to host
			testVectorFloat.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual(baseVectorFloat[i], (float)testVectorFloat[i]);
			}
		}

		//Test copy constructor (device)
		TEST_METHOD(CopyConstructor) {
			sharedVector<float> testVectorCopy = testVectorFloat;

			//Baseline vector test
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual((float)testVectorCopy[i], (float)testVectorFloat[i]);
				testVectorFloat[i] = 0.0f;
				Assert::AreNotEqual((float)testVectorCopy[i], (float)testVectorFloat[i]);
			}

			//Overwrite test
			testVectorFloat = testVectorCopy;
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual((float)testVectorCopy[i], (float)testVectorFloat[i]);
				testVectorCopy[i] = 0.0f;
				Assert::AreNotEqual((float)testVectorCopy[i], (float)testVectorFloat[i]);
			}

			//Render to test copy of device data
			cudaVector<float> deviceTest = testVectorFloat;

			//Copy again
			testVectorCopy = testVectorFloat;

			//Sync and compare
			testVectorCopy.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual((float)testVectorCopy[i], (float)testVectorFloat[i]);
			}
		}

		//Test move constructor (device)
		TEST_METHOD(MoveConstructor) {
			//Send to device
			cudaVector<float> deviceTest = testVectorFloat;

			//Self assignment
			sharedVector<float> testVectorMove = move(testVectorFloat);

			//Test that old vector no longer holds data
			Assert::AreEqual((uint)testVectorFloat.size(), 0u);
			Assert::AreEqual(testVectorFloat.core.size(), 0u);

			//Sync back to host
			testVectorMove.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual(baseVectorFloat[i], (float)testVectorMove[i]);
			}
		}

		//Test destructor (device)
		//Ignored because it cannot be multithreaded
		BEGIN_TEST_METHOD_ATTRIBUTE(Destructor)
			TEST_IGNORE()
		END_TEST_METHOD_ATTRIBUTE()
		TEST_METHOD(Destructor) {
			size_t freeMemBefore, freeMemAfterCopy, freeMemAfterDestroy, totalMem;
			cudaMemGetInfo(&freeMemBefore, &totalMem);
			{
				default_random_engine generator((unsigned int)time(NULL));
				uniform_real<float> distribution;
				vector<float> baseVector;
				sharedVector<float> testVector;
				int testItems = 5;

				for (int i = 0; i < testItems; i++) {
					float newVal = distribution(generator);
					baseVector.push_back(newVal);
					testVector.push_back(newVal);
				}

				//Send to device
				cudaVector<float> deviceTest = testVector;
				cudaMemGetInfo(&freeMemAfterCopy, &totalMem);

				Assert::AreNotEqual(freeMemBefore, freeMemAfterCopy);

				//Sync back to host
				testVector.sync();
				for (int i = 0; i < testItems; i++) {
					Assert::AreEqual(baseVector[i], (float)testVector[i]);
				}
			}
			cudaMemGetInfo(&freeMemAfterDestroy, &totalMem);
			Assert::AreEqual(freeMemBefore, freeMemAfterDestroy);
		}

		//Test 2D vector (device)
		TEST_METHOD(Vector2D) {
			default_random_engine generator((unsigned int)time(NULL));
			uniform_real<float> distribution;

			sharedVector<sharedVector<float> > testVector;
			vector<vector<float> > baseVector;

			//Render on no elements
			cudaVector<cudaVector<float> > deviceTest = testVector;
			Assert::AreEqual(deviceTest.size(), 0u);

			int testItems = 5;

			for (int dim = 0; dim < 2; dim++) {
				sharedVector<float> innerTest;
				vector<float> innerBase;
				for (int i = 0; i < testItems; i++) {
					float newVal = distribution(generator);
					innerBase.push_back(newVal);
					innerTest.push_back(newVal);
				}

				testVector.push_back(innerTest);
				baseVector.push_back(innerBase);
			}

			//Send to device
			deviceTest = testVector;

			//Test host equivalency
			for (int dim = 0; dim < 2; dim++) {
				for (int i = 0; i < testItems; i++) {
					Assert::AreEqual(baseVector[dim][i], (float)testVector[dim][i]);
					testVector[dim][i] = distribution(generator);
					Assert::AreNotEqual(baseVector[dim][i], (float)testVector[dim][i]);
				}
			}

			//Sync back to host
			testVector.sync();
			for (int dim = 0; dim < 2; dim++) {
				for (int i = 0; i < testItems; i++) {
					Assert::AreEqual(baseVector[dim][i], (float)testVector[dim][i]);
				}
			}

			//Copy to second 2D vector
			sharedVector<sharedVector<float> > testVectorCopy = testVector;
			for (int dim = 0; dim < 2; dim++) {
				for (int i = 0; i < testItems; i++) {
					Assert::AreEqual(baseVector[dim][i], (float)testVector[dim][i]);
					Assert::AreEqual(baseVector[dim][i], (float)testVectorCopy[dim][i]);
				}
			}

			//Move to second 2D vector
			sharedVector<sharedVector<float> > testVectorMove = move(testVector);
			for (int dim = 0; dim < 2; dim++) {
				for (int i = 0; i < testItems; i++) {
					Assert::AreEqual(baseVector[dim][i], (float)testVectorMove[dim][i]);
				}
			}
			Assert::AreEqual((int)testVector.size(), 0);
		}

		//Test 3D vector (device)
		TEST_METHOD(Vector3D) {
			default_random_engine generator((unsigned int)time(NULL));
			uniform_real<float> distribution;

			sharedVector<sharedVector<sharedVector<float> > > testVector;
			vector<vector<vector<float> > > baseVector;

			//Render on no elements
			cudaVector<cudaVector<cudaVector<float> > > deviceTest = testVector;
			Assert::AreEqual(deviceTest.size(), 0u);

			int testItems = 5;
			int testDims = 2;
			int testOuterDims = 2;

			for (int outerDim = 0; outerDim < testOuterDims; outerDim++) {
				sharedVector<sharedVector<float> > innerTest;
				vector<vector<float> > innerBase;
				for (int dim = 0; dim < testDims; dim++) {
					sharedVector<float> innerestTest;
					vector<float> innerestBase;
					for (int i = 0; i < testItems; i++) {
						float newVal = distribution(generator);
						innerestBase.push_back(newVal);
						innerestTest.push_back(newVal);
					}

					innerTest.push_back(innerestTest);
					innerBase.push_back(innerestBase);
				}
				testVector.push_back(innerTest);
				baseVector.push_back(innerBase);
			}

			//Send to device
			deviceTest = testVector;

			//Test host equivalency
			for (int outerDim = 0; outerDim < testOuterDims; outerDim++) {
				for (int dim = 0; dim < testDims; dim++) {
					for (int i = 0; i < testItems; i++) {
						Assert::AreEqual(baseVector[outerDim][dim][i], (float)testVector[outerDim][dim][i]);
						testVector[outerDim][dim][i] = distribution(generator);
						Assert::AreNotEqual(baseVector[outerDim][dim][i], (float)testVector[outerDim][dim][i]);
					}
				}
			}

			//Sync back to host
			testVector.sync();
			for (int outerDim = 0; outerDim < testOuterDims; outerDim++) {
				for (int dim = 0; dim < testDims; dim++) {
					for (int i = 0; i < testItems; i++) {
						Assert::AreEqual(baseVector[outerDim][dim][i], (float)testVector[outerDim][dim][i]);
					}
				}
			}

			//Copy to second 2D vector
			sharedVector<sharedVector<sharedVector<float> > > testVectorCopy = testVector;
			for (int outerDim = 0; outerDim < testOuterDims; outerDim++) {
				for (int dim = 0; dim < testDims; dim++) {
					for (int i = 0; i < testItems; i++) {
						Assert::AreEqual(baseVector[outerDim][dim][i], (float)testVector[outerDim][dim][i]);
						Assert::AreEqual(baseVector[outerDim][dim][i], (float)testVectorCopy[outerDim][dim][i]);
					}
				}
			}

			//Move to second 2D vector
			sharedVector<sharedVector<sharedVector<float> > > testVectorMove = move(testVector);
			for (int outerDim = 0; outerDim < testOuterDims; outerDim++) {
				for (int dim = 0; dim < testDims; dim++) {
					for (int i = 0; i < testItems; i++) {
						Assert::AreEqual(baseVector[outerDim][dim][i], (float)testVectorMove[outerDim][dim][i]);
					}
				}
			}
			Assert::AreEqual((int)testVector.size(), 0);
		}

		//Test class with vectors (device)
		TEST_METHOD(MixedClass) {
			default_random_engine generator((unsigned int)time(NULL));
			uniform_int<int> distribution;
			uniform_real<float> distributionF;

			class TestClassCore {
			public:
				int testInt;
				int testFloat;
				cudaVector<int> testIntVec;
				cudaVector<float> testFloatVec;
			};

			class TestClass : public cuClass<TestClassCore> {
			public:
				sharedVector<int> testIntVec;
				sharedVector<float> testFloatVec;

				TestClassCore& render(bool device) override {
					core.testIntVec = testIntVec.render(device);
					core.testFloatVec = testFloatVec.render(device);
					return core;
				}
				void sync() override {
					testIntVec.sync();
					testFloatVec.sync();
				}
			};

			TestClass test;
			vector<int> baseVectorInt;
			vector<float> baseVectorFloat;

			TestClassCore renderedTest = test;
			Assert::AreEqual(renderedTest.testIntVec.size(), 0u);
			Assert::AreEqual(renderedTest.testFloatVec.size(), 0u);

			int testItems = 5;

			for (int i = 0; i < testItems; i++) {
				int newVal = distribution(generator);
				baseVectorInt.push_back(newVal);
				test.testIntVec.push_back(newVal);
				float newValF = distributionF(generator);
				test.testFloatVec.push_back(newValF);
				baseVectorFloat.push_back(newValF);
			}

			//Send to device
			renderedTest = test;

			//Sync back to host
			test.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual(baseVectorInt[i], (int)test.testIntVec[i]);
				Assert::AreEqual(baseVectorFloat[i], (float)test.testFloatVec[i]);
			}
		}

		//Change vector size while data is pushed (device)
		TEST_METHOD(SizeChangedException)
		{
			//Send to device
			cudaVector<float> deviceTest = testVectorFloat;

			testVectorFloat.push_back(0.0f);

			//Sync back to host, should throw exception
			try {
				testVectorFloat.sync();
				Assert::Fail();
			}
			catch (sizeChangedException) {}
		}
	};

	TEST_CLASS(HostConstructors)
	{
	public:
		default_random_engine generator = default_random_engine((unsigned int)time(NULL));

		uniform_int<int> intDistribution;
		vector<int> baseVectorInt;
		sharedVector<int> testVectorInt;

		uniform_real<float> floatDistribution;
		vector<float> baseVectorFloat;
		sharedVector<float> testVectorFloat;
		int testItems = 5;

		TEST_METHOD_INITIALIZE(init) {
			for (int i = 0; i < testItems; i++) {
				int newVal = intDistribution(generator);
				baseVectorInt.push_back(newVal);
				testVectorInt.push_back(newVal);

				float newValFloat = floatDistribution(generator);
				baseVectorFloat.push_back(newValFloat);
				testVectorFloat.push_back(newValFloat);
			}
		}

		//Test basic render function (host)
		TEST_METHOD(RenderInt)
		{
			//Send to device
			cudaVector<int> hostTest = testVectorInt.render(false);

			//Sync back to host
			testVectorInt.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual(baseVectorInt[i], (int)testVectorInt[i]);
			}
		}

		//Test render and syncs for floats
		TEST_METHOD(RenderFloat)
		{
			//Send to device
			cudaVector<float> hostTest = testVectorFloat.render(false);

			//Sync back to host
			testVectorFloat.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual(baseVectorFloat[i], (float)testVectorFloat[i]);
			}
		}

		//Test quoted size (host)
		TEST_METHOD(TestSizeQuote) {
			sharedVector<float> testVector;

			Assert::AreEqual(sizeof(cudaVector<float>), testVector.byteSize());

			testVector.push_back(0.0f);

			Assert::AreEqual(sizeof(cudaVector<float>), testVector.byteSize());
		}

		//Self copy test (host)
		TEST_METHOD(SelfAssignment)
		{
			//Send to device
			cudaVector<float> hostTest = testVectorFloat.render(false);

			//Self assignment
			testVectorFloat = testVectorFloat;

			//Sync back to host
			testVectorFloat.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual(baseVectorFloat[i], (float)testVectorFloat[i]);
			}
		}

		//Test copy constructor (host)
		TEST_METHOD(CopyConstructor) {
			sharedVector<float> testVectorCopy = testVectorFloat;

			//Baseline vector test
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual((float)testVectorCopy[i], (float)testVectorFloat[i]);
				testVectorFloat[i] = 0.0f;
				Assert::AreNotEqual((float)testVectorCopy[i], (float)testVectorFloat[i]);
			}

			//Overwrite test
			testVectorFloat = testVectorCopy;
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual((float)testVectorCopy[i], (float)testVectorFloat[i]);
				testVectorCopy[i] = 0.0f;
				Assert::AreNotEqual((float)testVectorCopy[i], (float)testVectorFloat[i]);
			}

			//Render to test copy of device data
			cudaVector<float> hostTest = testVectorFloat.render(false);

			//Copy again
			testVectorCopy = testVectorFloat;

			//Sync and compare
			testVectorCopy.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual((float)testVectorCopy[i], (float)testVectorFloat[i]);
			}
		}

		//Test move constructor (host)
		TEST_METHOD(MoveConstructor) {
			//Send to device
			cudaVector<float> hostTest = testVectorFloat.render(false);

			//Self assignment
			sharedVector<float> testVectorMove = move(testVectorFloat);

			//Test that old vector no longer holds data
			Assert::AreEqual((uint)testVectorFloat.size(), 0u);
			Assert::AreEqual(testVectorFloat.core.size(), 0u);

			//Sync back to host
			testVectorMove.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual(baseVectorFloat[i], (float)testVectorMove[i]);
			}
		}

		//Test destructor (host)
		//Ignored because it cannot be multithreaded
		BEGIN_TEST_METHOD_ATTRIBUTE(Destructor)
			TEST_IGNORE()
			END_TEST_METHOD_ATTRIBUTE()
		TEST_METHOD(Destructor) {
			size_t freeMemBefore, freeMemAfterCopy, freeMemAfterDestroy, totalMem;
			cudaMemGetInfo(&freeMemBefore, &totalMem);
			{
				default_random_engine generator((unsigned int)time(NULL));
				uniform_real<float> distribution;
				vector<float> baseVector;
				sharedVector<float> testVector;
				int testItems = 5;

				for (int i = 0; i < testItems; i++) {
					float newVal = distribution(generator);
					baseVector.push_back(newVal);
					testVector.push_back(newVal);
				}

				//Send to host
				cudaVector<float> hostTest = testVector.render(false);
				cudaMemGetInfo(&freeMemAfterCopy, &totalMem);

				Assert::AreNotEqual(freeMemBefore, freeMemAfterCopy);

				//Sync back to host
				testVector.sync();
				for (int i = 0; i < testItems; i++) {
					Assert::AreEqual(baseVector[i], (float)testVector[i]);
				}
			}
			cudaMemGetInfo(&freeMemAfterDestroy, &totalMem);
			Assert::AreEqual(freeMemBefore, freeMemAfterDestroy);
		}

		//Test 2D vector (host)
		TEST_METHOD(Vector2D) {
			default_random_engine generator((unsigned int)time(NULL));
			uniform_real<float> distribution;

			sharedVector<sharedVector<float> > testVector;
			vector<vector<float> > baseVector;

			//Render on no elements
			cudaVector<cudaVector<float> > hostTest = testVector.render(false);
			Assert::AreEqual(hostTest.size(), 0u);

			int testItems = 5;

			for (int dim = 0; dim < 2; dim++) {
				sharedVector<float> innerTest;
				vector<float> innerBase;
				for (int i = 0; i < testItems; i++) {
					float newVal = distribution(generator);
					innerBase.push_back(newVal);
					innerTest.push_back(newVal);
				}

				testVector.push_back(innerTest);
				baseVector.push_back(innerBase);
			}

			//Send to device
			hostTest = testVector.render(false);

			//Test host equivalency
			for (int dim = 0; dim < 2; dim++) {
				for (int i = 0; i < testItems; i++) {
					Assert::AreEqual(baseVector[dim][i], (float)testVector[dim][i]);
					testVector[dim][i] = distribution(generator);
					Assert::AreNotEqual(baseVector[dim][i], (float)testVector[dim][i]);
				}
			}

			//Sync back to host
			testVector.sync();
			for (int dim = 0; dim < 2; dim++) {
				for (int i = 0; i < testItems; i++) {
					Assert::AreEqual(baseVector[dim][i], (float)testVector[dim][i]);
				}
			}

			//Copy to second 2D vector
			sharedVector<sharedVector<float> > testVectorCopy = testVector;
			for (int dim = 0; dim < 2; dim++) {
				for (int i = 0; i < testItems; i++) {
					Assert::AreEqual(baseVector[dim][i], (float)testVector[dim][i]);
					Assert::AreEqual(baseVector[dim][i], (float)testVectorCopy[dim][i]);
				}
			}

			//Move to second 2D vector
			sharedVector<sharedVector<float> > testVectorMove = move(testVector);
			for (int dim = 0; dim < 2; dim++) {
				for (int i = 0; i < testItems; i++) {
					Assert::AreEqual(baseVector[dim][i], (float)testVectorMove[dim][i]);
				}
			}
			Assert::AreEqual((int)testVector.size(), 0);
		}

		//Test 3D vector (host)
		TEST_METHOD(Vector3D) {
			default_random_engine generator((unsigned int)time(NULL));
			uniform_real<float> distribution;

			sharedVector<sharedVector<sharedVector<float> > > testVector;
			vector<vector<vector<float> > > baseVector;

			//Render on no elements
			cudaVector<cudaVector<cudaVector<float> > > hostTest = testVector.render(false);
			Assert::AreEqual(hostTest.size(), 0u);

			int testItems = 5;
			int testDims = 2;
			int testOuterDims = 2;

			for (int outerDim = 0; outerDim < testOuterDims; outerDim++) {
				sharedVector<sharedVector<float> > innerTest;
				vector<vector<float> > innerBase;
				for (int dim = 0; dim < testDims; dim++) {
					sharedVector<float> innerestTest;
					vector<float> innerestBase;
					for (int i = 0; i < testItems; i++) {
						float newVal = distribution(generator);
						innerestBase.push_back(newVal);
						innerestTest.push_back(newVal);
					}

					innerTest.push_back(innerestTest);
					innerBase.push_back(innerestBase);
				}
				testVector.push_back(innerTest);
				baseVector.push_back(innerBase);
			}

			//Send to device
			hostTest = testVector.render(false);

			//Test host equivalency
			for (int outerDim = 0; outerDim < testOuterDims; outerDim++) {
				for (int dim = 0; dim < testDims; dim++) {
					for (int i = 0; i < testItems; i++) {
						Assert::AreEqual(baseVector[outerDim][dim][i], (float)testVector[outerDim][dim][i]);
						testVector[outerDim][dim][i] = distribution(generator);
						Assert::AreNotEqual(baseVector[outerDim][dim][i], (float)testVector[outerDim][dim][i]);
					}
				}
			}

			//Sync back to host
			testVector.sync();
			for (int outerDim = 0; outerDim < testOuterDims; outerDim++) {
				for (int dim = 0; dim < testDims; dim++) {
					for (int i = 0; i < testItems; i++) {
						Assert::AreEqual(baseVector[outerDim][dim][i], (float)testVector[outerDim][dim][i]);
					}
				}
			}

			//Copy to second 2D vector
			sharedVector<sharedVector<sharedVector<float> > > testVectorCopy = testVector;
			for (int outerDim = 0; outerDim < testOuterDims; outerDim++) {
				for (int dim = 0; dim < testDims; dim++) {
					for (int i = 0; i < testItems; i++) {
						Assert::AreEqual(baseVector[outerDim][dim][i], (float)testVector[outerDim][dim][i]);
						Assert::AreEqual(baseVector[outerDim][dim][i], (float)testVectorCopy[outerDim][dim][i]);
					}
				}
			}

			//Move to second 2D vector
			sharedVector<sharedVector<sharedVector<float> > > testVectorMove = move(testVector);
			for (int outerDim = 0; outerDim < testOuterDims; outerDim++) {
				for (int dim = 0; dim < testDims; dim++) {
					for (int i = 0; i < testItems; i++) {
						Assert::AreEqual(baseVector[outerDim][dim][i], (float)testVectorMove[outerDim][dim][i]);
					}
				}
			}
			Assert::AreEqual((int)testVector.size(), 0);
		}

		//Test class with vectors (host)
		TEST_METHOD(MixedClass) {
			default_random_engine generator((unsigned int)time(NULL));
			uniform_int<int> distribution;
			uniform_real<float> distributionF;

			class TestClassCore {
			public:
				int testInt;
				int testFloat;
				cudaVector<int> testIntVec;
				cudaVector<float> testFloatVec;
			};

			class TestClass : public cuClass<TestClassCore> {
			public:
				sharedVector<int> testIntVec;
				sharedVector<float> testFloatVec;

				TestClassCore& render(bool device) override {
					core.testIntVec = testIntVec.render(device);
					core.testFloatVec = testFloatVec.render(device);
					return core;
				}
				void sync() override {
					testIntVec.sync();
					testFloatVec.sync();
				}
			};

			TestClass test;
			vector<int> baseVectorInt;
			vector<float> baseVectorFloat;

			TestClassCore renderedTest = test.render(false);
			Assert::AreEqual(renderedTest.testIntVec.size(), 0u);
			Assert::AreEqual(renderedTest.testFloatVec.size(), 0u);

			int testItems = 5;

			for (int i = 0; i < testItems; i++) {
				int newVal = distribution(generator);
				baseVectorInt.push_back(newVal);
				test.testIntVec.push_back(newVal);
				float newValF = distributionF(generator);
				test.testFloatVec.push_back(newValF);
				baseVectorFloat.push_back(newValF);
			}

			//Send to device
			renderedTest = test.render(false);

			//Sync back to host
			test.sync();
			for (int i = 0; i < testItems; i++) {
				Assert::AreEqual(baseVectorInt[i], (int)test.testIntVec[i]);
				Assert::AreEqual(baseVectorFloat[i], (float)test.testFloatVec[i]);
			}
		}

		//Change vector size while data is pushed (host)
		TEST_METHOD(SizeChangedException)
		{
			default_random_engine generator((unsigned int)time(NULL));
			uniform_real<float> distribution;
			sharedVector<float> testVector;
			int testItems = 5;

			for (int i = 0; i < testItems; i++) {
				float newVal = distribution(generator);
				testVector.push_back(newVal);
			}

			//Send to device
			cudaVector<float> hostTest = testVector.render(false);

			testVector.push_back(0.0f);

			//Sync back to host, should throw exception
			try {
				testVector.sync();
				Assert::Fail();
			}
			catch (sizeChangedException) {}
		}
	};


	TEST_CLASS(TrasientVectorUsage) {
		sharedVector<int> testVectorInt;

		uint testItems = 5;

		TEST_METHOD_INITIALIZE(init) {
			vector<uint> sizes({ testItems });
			testVectorInt.transientResize(sizes, (uint)sizes.size());
		}

		//Tranisent data shall act as a placeholder in CPU memory and not take up any space
		TEST_METHOD(TransientRender) {
			Assert::AreEqual(0, (int)testVectorInt.size());

			cudaVector<int> deviceTest = testVectorInt.render(true);

			Assert::AreEqual(testItems, deviceTest.size());
		}

		TEST_METHOD(TransientRender2D) {
			sharedVector<sharedVector<int> > testVectorInt2D;
			uint testDepth = 10;

			vector<uint> sizes({ testItems, testDepth });
			testVectorInt2D.transientResize(sizes, (uint)sizes.size());

			Assert::AreEqual(testDepth, (uint)testVectorInt2D.size());
			for (uint i = 0; i < testVectorInt2D.size(); i++) {
				Assert::AreEqual(0u, (uint)testVectorInt2D[i].size());
			}

			cudaVector<cudaVector<int> > deviceTest = testVectorInt2D.render(false);

			Assert::AreEqual(testDepth, deviceTest.size());
			for (uint i = 0; i < deviceTest.size(); i++) {
				Assert::AreEqual(testItems, deviceTest[i].size());
			}
			Assert::AreNotEqual(deviceTest[0].d_ptr, deviceTest[deviceTest.size() - 1].d_ptr);
		}

		//Currently do not have a way to test with the recursive method
		/*
		//Test too many arguments
		TEST_METHOD(TooManyArguments) {
			sharedVector<sharedVector<int> > testVectorInt2D;
			uint testDepth = 10;

			vector<uint> sizes({ testItems, testDepth, 10 });

			try {
				testVectorInt2D.transientResize(sizes, (uint)sizes.size());
				Assert::Fail();
			}
			catch (invalidTransientVector()) {}

			try {
				testVectorInt.transientResize(sizes, (uint)sizes.size());
				Assert::Fail();
			}
			catch (invalidTransientVector()) {}
		}

		//Test not enough arguments
		TEST_METHOD(TooFewArguments) {
			sharedVector<sharedVector<int> > testVectorInt2D;
			uint testDepth = 10;

			vector<uint> sizes({ testItems });

			try {
				testVectorInt2D.transientResize(sizes, (uint)sizes.size());
				Assert::Fail();
			}
			catch (invalidTransientVector()) {}

			sizes.clear();

			try {
				testVectorInt2D.transientResize(sizes, (uint)sizes.size());
				Assert::Fail();
			}
			catch (invalidTransientVector()) {}
		}*/
	};

	TEST_CLASS(AddAndRemoveItems) {
		sharedVector<uint> testVector;

		//Test add item just pushes
		TEST_METHOD(AddItem) {
			testVector.addItem(1);
			testVector.addItem(2);
			testVector.addItem(3);

			Assert::AreEqual((int)testVector.size(), 3);
			Assert::AreEqual((int)testVector[0], 1);
			Assert::AreEqual((int)testVector[1], 2);
			Assert::AreEqual((int)testVector[2], 3);
		}

		//Test delete item does not change size
		TEST_METHOD(DeleteItem) {
			testVector.addItem(1);
			testVector.addItem(2);
			testVector.addItem(3);
			testVector.deleteItem(0);

			Assert::AreEqual((int)testVector.size(), 3);
			Assert::AreEqual((int)testVector[0], 1);
			Assert::AreEqual((int)testVector[1], 2);
			Assert::AreEqual((int)testVector[2], 3);
		}

		//Delete item outside range, Delete item already deleted
		TEST_METHOD(InvalidDeletions) {
			testVector.addItem(1);
			testVector.addItem(2);
			testVector.addItem(3);

			try {
				testVector.deleteItem(3);
				Assert::Fail();
			}
			catch (invalidDeletionLocation) {}

			try {
				testVector.deleteItem(0);
				testVector.deleteItem(0);
				Assert::Fail();
			}
			catch (invalidDeletionLocation) {}

		}

		//Test delete->add does not change size, but changes contents
		TEST_METHOD(DeleteThenAdd) {
			testVector.addItem(1);
			testVector.addItem(2);
			testVector.addItem(3);
			testVector.deleteItem(0);
			testVector.addItem(4);

			Assert::AreEqual((int)testVector.size(), 3);
			Assert::AreEqual((int)testVector[0], 4);
			Assert::AreEqual((int)testVector[1], 2);
			Assert::AreEqual((int)testVector[2], 3);
		}

		//Add after delete->add
		TEST_METHOD(DeleteThenAddAdd) {
			testVector.addItem(1);
			testVector.addItem(2);
			testVector.deleteItem(0);
			testVector.addItem(4);
			testVector.addItem(3);

			Assert::AreEqual((int)testVector.size(), 3);
			Assert::AreEqual((int)testVector[0], 4);
			Assert::AreEqual((int)testVector[1], 2);
			Assert::AreEqual((int)testVector[2], 3);
		}

		//Effective size functionality
		TEST_METHOD(EffectiveSize) {
			testVector.addItem(1);
			testVector.addItem(2);
			Assert::AreEqual(testVector.effectiveSize(), 2);

			testVector.deleteItem(0);
			Assert::AreEqual(testVector.effectiveSize(), 1);

			testVector.addItem(3);
			Assert::AreEqual(testVector.effectiveSize(), 2);
		}

		//Clear after deleted has data
		TEST_METHOD(Clear) {
			testVector.addItem(1);
			testVector.addItem(2);
			testVector.deleteItem(0);
			testVector.clear();
			Assert::AreEqual(testVector.effectiveSize(), 0);

			testVector.addItem(4);
			testVector.addItem(3);
			Assert::AreEqual(testVector.effectiveSize(), 2);
		}
	};
}