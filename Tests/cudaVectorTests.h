#include "../LinearBackprop/sharedVector/cudaVector.h"

void sumTo0(cudaVector<int>& vec);
void increment(cudaVector<int>& vec);
void overflow(cudaVector<int>& vec);
uint getSize(cudaVector<int>& vec);