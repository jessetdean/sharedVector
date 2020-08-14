#pragma once

#include "cudaIncludes.h"
#include "cuClass.h"
#include "cudaVector.h"
#include "exceptions.h"

#pragma warning(disable:4251)

template < template <typename...> class base, typename derived>
struct is_base_of_template_impl
{
	template<typename... Ts>
	static constexpr std::true_type  test(const base<Ts...>*);

	static constexpr std::false_type test(...);
	using type = decltype(test(std::declval<derived*>()));
};

template < template <typename...> class base, typename derived>
using is_base_of_template = typename is_base_of_template_impl<base, derived>::type;

//Base level shared vectors, used for standard data and structs
template<typename T, bool = is_base_of_template<cuClass, T>::value>
class DLL_NETWORK sharedVector : public std::vector<T>, public cuClass<cudaVector<T> > {
public:
	//Dirty bit, skip rendering if false
	typedef T baseType;

	//Swap all data-containing elements
	void swap(sharedVector<T>& first, sharedVector<T>& second) {
		using std::swap;
		swap(first.alloc, second.alloc);
		swap(first.isDevice, second.isDevice);
		swap(first.core, second.core);
		swap(static_cast<std::vector<T>&>(first), static_cast<std::vector<T>&>(second));
	}

	///Rule of five for transfers and memory management

	//Standard constructor for 0 initialization and empty vector
	sharedVector() = default;

	//Copy over std vector to contained vector
	sharedVector(std::vector<T> init) {
		static_cast<std::vector<T>&>(*this) = init;
	}

	//Copy constructor
	sharedVector(const sharedVector & other) : std::vector<T>(other) {
		core = other.core;
		alloc = other.alloc;
		isDevice = other.isDevice;
		if (isDevice) {
			if (alloc > 0) {
				cudaMalloc((void**)&core.d_ptr, alloc * sizeof(T));
			}
			else core.d_ptr = nullptr;
			if (core.size() > 0) {
				cudaMemcpyAsync(core.d_ptr, other.core.d_ptr, core.size() * sizeof(T), cudaMemcpyDeviceToDevice);
			}
		}
		else {
			if (alloc > 0) {
				core.d_ptr = (T*)malloc(alloc * sizeof(T));
			}
			else core.d_ptr = nullptr;
			if (core.size() > 0) {
				memcpy(core.d_ptr, other.core.d_ptr, core.size() * sizeof(T));
			}
		}
	}

	//Move constructor
	sharedVector(sharedVector && other) noexcept {
		swap(*this, other);
		other.core.d_ptr = 0;
	}

	//Copy assignment
	sharedVector& operator=(const sharedVector & other) {
		return *this = sharedVector<T>(other);
	}

	//Move assignment
	sharedVector& operator=(sharedVector && other) noexcept {
		swap(*this, other);
		return *this;
	}

	//Destructor
	~sharedVector() {
		clearDevice();
	}

	/*
	 * Marks the shared vector for device only usage
	 * Parameter determines the size of the vector in the device
	 * Host side vector should not be used after this call
	 * Vector is ordered lowest dimension to highest
	 */
	void transientResize(vector<uint> & deviceSizes, uint order)  {
		//Only lowest dimension skips host allocation
		transient = true;
		transientSize = deviceSizes.front();
	}

	//Functions for shift-less additions and deletions
	int effectiveSize() {
		return (int)std::vector<T>::size() - (int)deleted.size();
	}

	void clearDevice() {
		if (alloc > 0 && core.d_ptr != nullptr) {
			if (isDevice) {
				cudaFree(core.d_ptr);
			}
			else {
				free(core.d_ptr);
			}
		}
		alloc = 0;
	}

	void clear() {
		std::vector<T>::clear();
		deleted.clear();
		clearDevice();
	}

	void deleteItem(uint index) {
		//dirty = true;//Not needed, doesn't change anything about the payload vector

		if (index >= std::vector<T>::size() || std::count(deleted.begin(), deleted.end(), index) > 0)
			throw invalidDeletionLocation();
		deleted.push_back(index);
	}

	int addItem(T item) {
		dirty = true;

		int newID;
		if (deleted.empty()) {
			newID = (int)std::vector<T>::size();
			push_back(item);
		}
		else {
			newID = deleted.back();
			deleted.pop_back();
			(*this)[newID] = item;
		}
		return newID;
	}

	///cuClass overrides
	void reset() override {
		for (uint32_t i = 0; i < std::vector<T>::size(); i++) {
			(*this)[i] = {};
		}
	}

	size_t byteSize() override {
		//Only if single block
		//size_t totalSize = 0;
		//Get sizes from child elements

		return sizeof(core);
	}

	cudaVector<T>& render(bool device) override {
		if (alloc > 0 && device != isDevice)
			throw locationChangeException();
		isDevice = device;

		//Allocate memory for this item
		core._size = transient ? transientSize : (uint32_t)std::vector<T>::size();
		if (alloc < core.size()) {
			dirty = true;

			if (alloc == 0) {
				alloc = core.size();
			}
			else {
				alloc *= 2;
				if (alloc < core.size()) alloc = core.size();
				if (isDevice) {
					cudaFree(core.d_ptr);
				}
				else {
					free(core.d_ptr);
				}
			}
			if (isDevice) {
				cudaMalloc((void**)&core.d_ptr, alloc * sizeof(T));
			}
			else {
				core.d_ptr = (T*)malloc(alloc * sizeof(T));
			}
		}

		if (core.d_ptr == 0)
			return core;

		if (!transient && dirty) {
			dirty = false;

			if (isDevice) {
				cudaMemcpyAsync(core.d_ptr, std::vector<T>::data(), core.size() * sizeof(T), cudaMemcpyHostToDevice);
			}
			else {
				memcpy(core.d_ptr, std::vector<T>::data(), core.size() * sizeof(T));
			}
		}

		return core;
	}

	void sync() override {
		if (core.size() != std::vector<T>::size())
			throw sizeChangedException();

		//Sync main data
		if (isDevice) {
			cudaMemcpy(std::vector<T>::data(), core.d_ptr, core.size() * sizeof(T), cudaMemcpyDeviceToHost);
		}
		else {
			memcpy(std::vector<T>::data(), core.d_ptr, core.size() * sizeof(T));
		}
	}

	void setDirty() override {
		dirty = true;
	}

private:
	bool dirty = true;
	bool transient = false;
	uint transientSize = 0;
	std::vector<uint> deleted;
	uint alloc = 0;
	bool isDevice = true;
};

//Higher order shared vectors for vectors of cuClasses, or shared vectors of shared vectors
template<typename T>
class DLL_NETWORK sharedVector<T, true> : public std::vector<T>, public cuClass<cudaVector<typename T::coreType> > {
public:
	//Dirty bit, skip rendering if false
	bool dirty = true;
	std::vector<typename T::coreType> renderedPointers;
	typedef typename T::coreType baseType;

	//Swap all data-containing elements
	void swap(sharedVector<T>& first, sharedVector<T>& second) {
		using std::swap;
		swap(first.alloc, second.alloc);
		swap(first.isDevice, second.isDevice);
		swap(first.core, second.core);
		swap(static_cast<std::vector<T>&>(first), static_cast<std::vector<T>&>(second));
	}

	///Rule of five for transfers and memory management

	//Standard constructor for 0 initialization and empty vector
	sharedVector() = default;

	//Copy over std vector to contained vector
	sharedVector(std::vector<T> init) {
		static_cast<std::vector<T>&>(*this) = init;
	}

	//Copy constructor
	sharedVector(const sharedVector& other) : std::vector<T>(other) {
		core = other.core;
		alloc = other.alloc;
		isDevice = other.isDevice;
		if (isDevice) {
			if (alloc > 0) {
				cudaMalloc((void**)&core.d_ptr, alloc * sizeof(T::coreType));
			}
			else core.d_ptr = nullptr;
			if (core.size() > 0) {
				cudaMemcpyAsync(core.d_ptr, other.core.d_ptr, core.size() * sizeof(T::coreType), cudaMemcpyDeviceToDevice);
			}
		}
		else {
			if (alloc > 0) {
				core.d_ptr = (T::coreType*)malloc(alloc * sizeof(T::coreType));
			}
			else core.d_ptr = nullptr;
			if (core.size() > 0) {
				memcpy(core.d_ptr, other.core.d_ptr, core.size() * sizeof(T::coreType));
			}
		}
	}

	//Move constructor
	sharedVector(sharedVector&& other) noexcept {
		swap(*this, other);
		other.core.d_ptr = 0;
	}

	//Copy assignment
	sharedVector& operator=(const sharedVector& other) {
		return *this = sharedVector<T>(other);
	}

	//Move assignment
	sharedVector& operator=(sharedVector&& other) noexcept {
		swap(*this, other);
		return *this;
	}

	//Destructor
	~sharedVector() {
		clearDevice();
	}

	/*
	Marks the shared vector for device only usage
	Parameter determines the size of the vector in the device
	Host side vector should not be used after this call
	Vector is ordered lowest dimension to highest
	*/
	void transientResize(std::vector<uint>& deviceSizes, uint order) {
		//Keep host record of all pointers for scope and later deletion
		std::vector<T>::resize(deviceSizes[order - 1]);

		//Recursively allocate and record
		for (int i = 0; i < std::vector<T>::size(); i++) {
			(*this)[i].transientResize(deviceSizes, order - 1);
		}
	}

	//Functions for shift-less additions and deletions
	int effectiveSize() {
		return (int)std::vector<T>::size() - (int)deleted.size();
	}

	void clearDevice() {
		if (alloc > 0 && core.d_ptr != nullptr) {
			if (isDevice) {
				cudaFree(core.d_ptr);
			}
			else {
				free(core.d_ptr);
			}
		}
		alloc = 0;
		for (uint32_t i = 0; i < std::vector<T>::size(); i++) {
			(*this)[i].clearDevice();
		}
	}

	void clear() {
		std::vector<T>::clear();
		deleted.clear();
		clearDevice();
	}

	void deleteItem(uint index) {
		//dirty = true;//Not needed, doesn't change anything about the payload vector

		if (index >= std::vector<T>::size() || std::count(deleted.begin(), deleted.end(), index) > 0)
			throw invalidDeletionLocation();
		deleted.push_back(index);
	}

	int addItem(T item) {
		dirty = true;

		int newID;
		if (deleted.empty()) {
			newID = (int)std::vector<T>::size();
			push_back(item);
		}
		else {
			newID = deleted.back();
			deleted.pop_back();
			(*this)[newID] = item;
		}
		return newID;
	}

	///cuClass overrides
	void reset() override {
		for (uint32_t i = 0; i < std::vector<T>::size(); i++) {
			(*this)[i].reset();
		}
	}

	size_t byteSize() override {
		//Only if single block
		//size_t totalSize = 0;
		//Get sizes from child elements

		return sizeof(core);
	}

	cudaVector<typename T::coreType>& render(bool device) override {
		if (alloc > 0 && device != isDevice)
			throw locationChangeException();
		isDevice = device;

		//Update size and check for dirty
		{
			uint newSize = transient ? transientSize : (uint32_t)std::vector<T>::size();
			dirty |= alloc < newSize;
			dirty |= newSize != core._size;
			core._size = newSize;
		}

		//Allocate memory for this item
		if (alloc < core.size()) {
			if (alloc == 0) {
				alloc = core.size();
			}
			else {
				alloc *= 2;
				if (alloc < core.size()) alloc = core.size();
				if (isDevice) {
					cudaFree(core.d_ptr);
				}
				else {
					free(core.d_ptr);
				}
			}
			if (isDevice) {
				cudaMalloc((void**)&core.d_ptr, alloc * sizeof(T::coreType));
			}
			else {
				core.d_ptr = (T::coreType*)malloc(alloc * sizeof(T::coreType));
			}
		}

		if (core.d_ptr == 0)
			return core;

		if (!transient && dirty) {
			dirty = false;

			//Update pointers for all items
			renderedPointers.resize(std::vector<T>::size());
			for (uint32_t i = 0; i < std::vector<T>::size(); i++) {
				renderedPointers[i] = (*this)[i].render(isDevice);
			}

			if (isDevice) {
				cudaMemcpyAsync(core.d_ptr, renderedPointers.data(), core.size() * sizeof(T::coreType), cudaMemcpyHostToDevice);
			}
			else {
				memcpy(core.d_ptr, renderedPointers.data(), core.size() * sizeof(T::coreType));
			}
		}

		return core;
	}

	void sync() override {
		if (core.size() != std::vector<T>::size())
			throw sizeChangedException();

		//Sync all sub-items
		for (uint32_t i = 0; i < (uint32_t)std::vector<T>::size(); i++) {
			(*this)[i].sync();
		}
	}

	void setDirty() override {
		dirty = true;

		for (uint32_t i = 0; i < (uint32_t)std::vector<T>::size(); i++) {
			(*this)[i].setDirty();
		}
	}

private:
	bool transient = false;
	uint transientSize = 0;
	std::vector<uint> deleted;
	uint alloc = 0;
	bool isDevice = true;
};

/*

	template<typename T>
	void sharedVectorBase<T>::merge(sharedVectorBase<T>& other) {
		reserve(vector<T>::size() + other.size());
		insert(vector<T>::end(), other.begin(), other.end());
	}
	*/

#pragma warning(default:4251)