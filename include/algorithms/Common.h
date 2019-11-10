/*
 * common.h
 *
 *  Created on: Nov 24, 2018
 *      Author: joseph
 */

#ifndef BC_ALGORITHM_COMMON_H_
#define BC_ALGORITHM_COMMON_H_

#ifdef BC_CPP17
#define BC_DEF_IF_CPP17(code) code
#else
#define BC_DEF_IF_CPP17(code)
#endif

#ifdef BC_CPP17
#ifndef BC_CPP17_EXECUTION
#define BC_CPP17_EXECUTION std::execution::par,
#endif
#endif

#include <iterator>
#include <vector>
namespace BC {
namespace algorithms {

template<class T>
struct ReferenceIterator: public std::vector<T*>::iterator {

	using iterator_category = std::random_access_iterator_tag;
	using parent = typename std::vector<T*>::iterator;

	ReferenceIterator(parent p) :
			parent(p) {
	}

	T& operator*() const {
		return *(parent::operator*());
	}

	T& operator[](std::size_t index) const {
		return *(parent::operator[](index));
	}
};

template<class T>
struct ReferenceList {

	std::vector<T*> container;

	template<class ... Ts>
	ReferenceList(Ts&... ts) :
			container { &ts... } {
	}

	auto begin() {
		return ReferenceIterator<T>(container.begin());
	}

	auto end() {
		return ReferenceIterator<T>(container.end());
	}

};

template<class T, class ... Ts>
ReferenceList<T> reference_list(T& t, Ts&... ts) {
	return ReferenceList<T>(t, ts...);
}

}
}

#endif /* COMMON_H_ */
