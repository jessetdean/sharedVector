#include "CppUnitTest.h"
#include <vector>
#include <random>
#include <time.h>
#include "cudaVectorTests.h"
#include "../LinearBackprop/sharedVector/sharedVector.h"

using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CudaVectorTest {
	TEST_CLASS(DeviceAccessFunctions)
	{
	public:
		default_random_engine generator = default_random_engine((unsigned int)time(NULL));
		uniform_int<int> distribution;

		vector<int> baseVector;
		sharedVector<int> testVector;
		int testItems = 5;

		TEST_METHOD_INITIALIZE(init) {
			for (int i = 0; i < testItems; i++) {
				int newVal = distribution(generator);
				baseVector.push_back(newVal);
				testVector.push_back(newVal);
			}
		}

		//Test information integrity on device side (random sum operation)
		TEST_METHOD(RandomSum)
		{
			//Send to device
			cudaVector<int> deviceTest = testVector;

			//Sum operation on rendered values
			sumTo0(deviceTest);

			//Sync back to host
			testVector.sync();

			//Ensure no device failutre
			cudaDeviceSynchronize();
			Assert::AreEqual((int)cudaSuccess, (int)cudaGetLastError());

			int sum = 0;
			for (int i = 0; i < testItems; i++)
				sum += baseVector[i];
			Assert::AreEqual(sum, (int)testVector[0]);
		}

		//Test device modification sync
		TEST_METHOD(RenderedModification)
		{
			//Send to device
			cudaVector<int> deviceTest = testVector;

			//Sum operation on rendered values
			increment(deviceTest);

			//Sync back to host
			testVector.sync();

			//Ensure no device failutre
			cudaDeviceSynchronize();
			Assert::AreEqual((int)cudaSuccess, (int)cudaGetLastError());

			for (int i = 0; i < testItems; i++)
				Assert::AreEqual(++baseVector[i], (int)testVector[i]);
		}

		//Test device size gathering
		TEST_METHOD(GetSize) {
			cudaVector<int> deviceTest = testVector;

			Assert::AreEqual((uint)testItems, getSize(deviceTest));
		}

		//Ignored because it's destructive to the device environment
		BEGIN_TEST_METHOD_ATTRIBUTE(OutOfBoundsErrorThrow)
			TEST_IGNORE()
		END_TEST_METHOD_ATTRIBUTE()
		//Test out of bounds error throwing
		TEST_METHOD(OutOfBoundsErrorThrow)
		{
			//Send to device
			cudaVector<int> deviceTest = testVector;
			//Overflow error
			overflow(deviceTest);
			cudaDeviceSynchronize();
			Assert::AreNotEqual((int)cudaSuccess, (int)cudaGetLastError());
			cudaDeviceReset();
		}
	};

	TEST_CLASS(HostAccessFunctions)
	{
	public:
		default_random_engine generator = default_random_engine((unsigned int)time(NULL));
		uniform_int<int> distribution;

		vector<int> baseVector;
		sharedVector<int> testVector;
		int testItems = 5;

		TEST_METHOD_INITIALIZE(init) {
			for (int i = 0; i < testItems; i++) {
				int newVal = distribution(generator);
				baseVector.push_back(newVal);
				testVector.push_back(newVal);
			}
		}

		//Test information integrity on host side (random sum operation)
		TEST_METHOD(RandomSum)
		{
			//Send to device
			cudaVector<int> deviceTest = testVector.render(false);

			//Sum operation on rendered values
			for (int i = 0; i < testItems; i++) {
				if (i != 0)
					deviceTest[0] += deviceTest[i];
			}

			//Sync back to host
			testVector.sync();

			int sum = 0;
			for (int i = 0; i < testItems; i++)
				sum += baseVector[i];
			Assert::AreEqual(sum, (int)testVector[0]);
		}

		//Test host modification sync
		TEST_METHOD(RenderedModification)
		{
			//Send to device
			cudaVector<int> deviceTest = testVector.render(false);

			//Sum operation on rendered values
			for (int i = 0; i < testItems; i++) {
				deviceTest[i] ++;
			}

			//Sync back to host
			testVector.sync();
			for (int i = 0; i < testItems; i++)
				Assert::AreEqual(++baseVector[i], (int)testVector[i]);
		}

		TEST_METHOD(DeviceFunctionality) {
			cudaVector<int> deviceTest = testVector.render(false);

			Assert::AreEqual((uint)testItems, deviceTest.size());
		}

		//Test out of bounds error throwing
		TEST_METHOD(OutOfBoundsErrorThrow)
		{
			//Send to device
			cudaVector<int> hostTest = testVector.render(false);

			//Force out of bounds
			try {
				int test = hostTest[testItems];
				Assert::Fail();
			}
			catch (outOfBounds e) {}
			try {
				int test = hostTest[-1];
				Assert::Fail();
			}
			catch (outOfBounds e) {}
		}
	};
}