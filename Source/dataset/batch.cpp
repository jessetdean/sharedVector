#include "batch.h"
#include "dataset.h"

Batch& cuBatch::render(bool device) {
	core.selections = selections.render(device);
	core.statistics = statistics.render(device);
	core.classResponses = classResponses.render(device);
	core.responses = responses.render(device);

	return core;
}

void cuBatch::sync() {
	statistics.sync();
	classResponses.sync();
	responses.sync();
}