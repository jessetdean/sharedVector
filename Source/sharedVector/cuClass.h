#pragma once

#include "cudaIncludes.h"
#include <vector>

/*
 * Interface for cuClass to enable calling functions on general types
 */
class ICuClass {
	virtual std::vector<ICuClass> getCuClasses() = 0;
	virtual void reset() = 0;
	virtual size_t byteSize() = 0;
	virtual void render(void * rendered, bool device = true) = 0;
	virtual void sync() = 0;
	virtual void clear() = 0;
	virtual void clearDevice() = 0;
	virtual void setDirty() = 0;
};

template<typename coreClass, typename... Ts>
class DLL_NETWORK cuClass : ICuClass {
public:
	coreClass core = {};

	typedef coreClass coreType;

	/*
	 * Rule of 5 with expected input as coreClass
	 */
	cuClass() = default;
	cuClass(const coreClass & init) { core = init; }
	cuClass(coreClass && init) { core = std::move(init); }
	void operator=(const coreClass & other) { core = other; }
	void operator=(coreClass && other) { core = std::move(other); }

	/*
	 * Stub for vector initialization
	 */
	virtual std::vector<ICuClass> getCuClasses() {
		return {};
	}

	/*
	 * Comparison operator
	 */
	friend bool operator==(const cuClass& l, const cuClass& r) { return l.core == r.core; }

	/*
	 * Reset all values to their default state
	 */
	virtual void reset() override { core = {}; }

	/*
	 * Recursively go through child vectors and return total memory size that would need
	 * to be allocated for a single block.
	 */
	virtual size_t byteSize() override { return sizeof(coreClass); }

	/*
	 * Handle the allocation and copying of any memory that this class will point to. It is 
	 * recommended to only point to other cuClasses so that they can in turn be rendered.
	 * Note that only segmented memory needs copying; local data should be handled in a 
	 * higher up function.
	 */
	virtual coreClass& render(bool device = true) { return core; }
	virtual void render(void * rendered, bool device = true) override { *(coreClass*)rendered = core; }
	operator coreClass() { return render(); }

	/*
	 * Handle copying back to host. This is only for child segmented memory, copies of this
	 * class's data is handled in a higher up function.
	 */
	virtual void sync() override {}

	/*
	 * Empty all contained shared vectors.
	 */
	virtual void clear() override {}

	/*
	 * Clear out renders to prepare for host/device switch
	 */
	virtual void clearDevice() override {}

	/*
	 * Meant to set the dirty bit on all cuda vectors and sub cuClasses
	 */
	virtual void setDirty() override {}
};