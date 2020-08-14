#include "datum.h"

Datum& cuDatum::render(bool device) {
	core.regressions = regressions.render(device);
	core.classes = classes.render(device);
	core.features = features.render(device);

	return core;
}

void cuDatum::clearDevice() {
	regressions.clearDevice();
	classes.clearDevice();
	features.clearDevice();
}