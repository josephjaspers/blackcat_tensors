#ifndef BLACKCAT_CPU_OPERATATIONS_H
#define BLACKCAT_CPU_OPERATATIONS_H

#include <vector>

namespace BC {

//Copy unsigned pointer
static void copy(unsigned* save, const unsigned* data, unsigned sz) {
	for (int i = 0; i < sz; ++i) {
		save[i] = data[i];
	}
}

//mult sum (inner product) -- used to calculate size from ranks
static void mult_sum(unsigned* save, const unsigned* base, const unsigned* data,
		unsigned data_sz) {
	*save = *base;
	for (int i = 0; i < data_sz; ++i) {
		*save *= data[i];
	}
}
//similair to above
static unsigned calc_sz(unsigned* ranks, unsigned order) {
	unsigned sz = 1;
	for (unsigned i = 0; i < order; ++i) {
		sz *= ranks[i];
	}
	return sz;
}
//initialize leading dimensions
static void init_leading_dimensions(unsigned* leading_dim, unsigned* dim,
		unsigned order) {

	unsigned base_sz = 1;
	for (int i = 0; i < order; ++i) {
		leading_dim[i] = base_sz;
		base_sz *= dim[i];
	}
}

//--------------------------unordered map hash stuff --------------------------------------------------//
//Shape equality for unordered map
struct shape_equality {
	bool operator()(const std::vector<unsigned>& s1, const std::vector<unsigned>& s2) const {
		if (s1.size() == s2.size())

			for (int i = 0; i < s1.size(); ++i) {
				if (s1[i] != s2[i]) {
					return false;
				}
			}
		else {
			return false;
		}
		return true;
	}
};
struct shape_hasher {
	size_t operator()(const std::vector<unsigned>& shape) const {
		size_t hash  = 0;
		for (int i = 0; i < shape.size(); ++i) {
			hash += shape[i] * pow(10, i);
		}
		return hash;
	}
};

struct indexShape_pairEqual {
	bool operator()(const std::pair<std::vector<unsigned>, std::vector<unsigned>>& s1, const std::pair<std::vector<unsigned>, std::vector<unsigned>>& s2) const {
		return s1.first == s2.first && s1.second == s2.second;
	}
};
struct indexShape_pairHasher {
	size_t operator()(const std::pair<std::vector<unsigned>, std::vector<unsigned>>& s1) const {
		size_t hash = 0;

		for (int i = 0; i < s1.first.size(); ++i) {
			hash += s1.first[i] * i;
		}
		for (int i = 0; i <s1.second.size(); ++i) {
			hash += s1.second[i] * i + s1.first.size();
		}
		return hash;
	}
};


}
;
#endif
