#pragma once

#include "sharedVector.h"
#include "datum.cuh"

#pragma warning(disable:4251)

class DLL_NETWORK cuDatum : public cuClass<Datum> {
public:
	sharedVector<float> regressions;
	sharedVector<uint> classes;
	sharedVector<float> features;

	//cuClass overrides
	Datum& render(bool device = true) override;
	void clearDevice() override;
};

#pragma warning(default:4251)