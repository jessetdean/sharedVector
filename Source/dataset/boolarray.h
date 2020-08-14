#pragma once

#ifndef BOOLARRAY_H_
#define BOOLARRAY_H_
#include <string.h>

#include <stdint.h>
#include <vector>
#include <cassert>

using namespace std;

static inline int __builtin_ctz(uint32_t x) {
	unsigned long ret;
	_BitScanForward(&ret, x);
	return (int)ret;
}

static inline int __builtin_ctzll(unsigned long long x) {
	unsigned long ret;
	_BitScanForward64(&ret, x);
	return (int)ret;
}

static inline int __builtin_ctzl(unsigned long x) {
	return sizeof(x) == 8 ? __builtin_ctzll(x) : __builtin_ctz((uint32_t)x);
}

static inline int numberOfTrailingZeros(uint64_t x) {
	if (x == 0) return 64;
	return  __builtin_ctzl((unsigned long)x);
}

class BoolArray {
public:

	vector<uint64_t> buffer;
	size_t sizeinbits;
	BoolArray(const size_t n, const uint64_t initval = 0) :
		buffer(n / 64 + (n % 64 == 0 ? 0 : 1), initval),
		sizeinbits(n) {
	}

	BoolArray() :
		buffer(), sizeinbits(0) {
	}

	BoolArray(const BoolArray& ba) :
		buffer(ba.buffer), sizeinbits(ba.sizeinbits) {
	}

	void setSizeInBits(const size_t sizeib) {
		sizeinbits = sizeib;
	}

	void toArray(vector<uint32_t>& ans) {
		size_t pos = 0;
		for (size_t k = 0; k < buffer.size(); ++k) {
			const uint64_t myword = buffer[k];
			for (int offset = 0; offset < 64; ++offset) {
				if ((myword >> offset) == 0) break;
				offset += numberOfTrailingZeros((myword >> offset));
				ans[pos++] = (unsigned int)(64 * k + offset);
				if (pos >= ans.size()) break;
			}
			if (pos >= ans.size()) break;
		}
	}


	BoolArray& operator=(const BoolArray& x) {
		this->buffer = x.buffer;
		this->sizeinbits = x.sizeinbits;
		return *this;
	}

	/**
	 * set to true (whether it was already set to true or not)
	 *
	 * This is an expensive (random access) API, you really ought to
	 * prepare a new word and then append it.
	 */
	inline void set(const size_t pos) {
		buffer[pos / 64] |= (static_cast<uint64_t> (1) << (pos
			% 64));
	}

	/**
	 * set to false (whether it was already set to false or not)
	 *
	 * This is an expensive (random access) API, you really ought to
	 * prepare a new word and then append it.
	 */
	inline void unset(const size_t pos) {
		buffer[pos / 64] |= ~(static_cast<uint64_t> (1) << (pos
			% 64));
	}

	/**
	 * true of false? (set or unset)
	 */
	inline bool get(const size_t pos) const {
		return (buffer[pos / 64] & (static_cast<uint64_t> (1) << (pos
			% 64))) != 0;
	}

	/**
	 * set all bits to 0
	 */
	void reset() {
		memset(&buffer[0], 0, sizeof(uint64_t) * buffer.size());
		sizeinbits = 0;
	}

	size_t sizeInBits() const {
		return sizeinbits;
	}

	~BoolArray() {
	}


};

#endif