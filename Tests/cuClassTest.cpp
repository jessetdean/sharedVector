#include "CppUnitTest.h"
#include <vector>
#include <random>
#include <time.h>
#include "../LinearBackprop/sharedVector/sharedVector.h"

using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CuClassTest {
	TEST_CLASS(Initialization) 
	{
	public:
		//Test compatibility with standard types
		TEST_METHOD(EmptyIntTest)
		{
			cuClass<int> empty;
			empty = 1;
			Assert::AreEqual((int)empty, 1);
		}

		TEST_METHOD(IntTest)
		{
			cuClass<int> constructed(1);
			Assert::AreEqual((int)constructed, 1);
		}

		TEST_METHOD(EmptyFloatTest)
		{
			cuClass<float> empty;
			empty = 1.0f;
			Assert::AreEqual((float)empty, 1.0f);
		}

		TEST_METHOD(FloatTest)
		{
			cuClass<float> constructed(1.0);
			Assert::AreEqual((float)constructed, 1.0f);
		}

		//Test compatibility with custom structs
		TEST_METHOD(EmptyStructTest)
		{
			struct TestStruct {
				int val = 1;
				float val2 = 2.0f;
			};
			cuClass<TestStruct> constructed;

			TestStruct output = constructed;
			Assert::AreEqual(output.val, 1);
			Assert::AreEqual(output.val2, 2.0f);
		}

		TEST_METHOD(StructTest)
		{
			struct TestStruct {
				int val = 1;
				float val2 = 2.0f;
			};
			cuClass<TestStruct> constructed({ 3, 4.0f });

			TestStruct output = constructed;
			Assert::AreEqual(output.val, 3);
			Assert::AreEqual(output.val2, 4.0f);
		}

		//Self copy test
		TEST_METHOD(SelfCopy)
		{
			cuClass<float> constructed(1.0);

			constructed = constructed;

			Assert::AreEqual((float)constructed, 1.0f);
		}

		//Equality operator test
		TEST_METHOD(EqualityOperator)
		{
			cuClass<float> constructed(1.0);
			cuClass<float> equal(1.0);
			cuClass<float> notEqual(0.0);

			Assert::AreEqual(true, constructed == equal);
			Assert::AreEqual(false, constructed == notEqual);
		}
	};
}