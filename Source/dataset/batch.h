#pragma once

#include "sharedVector.h"
#include "batch.cuh"
#include <random>

#pragma warning(disable:4251)

class DLL_NETWORK cuBatch : public cuClass<Batch> {
public:
	sharedVector<uint> selections;
	sharedVector<Statistics> statistics;
	sharedVector<sharedVector<Response> > classResponses;
	sharedVector<sharedVector<float> > responses;

	//cuClass overrides
	Batch& render(bool device = true) override;
	void sync() override;

	friend class cuDataset;

private:
	uint lastIndex = 0;
};

#pragma warning(default:4251)