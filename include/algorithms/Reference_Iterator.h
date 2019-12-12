/*
 * Reference_Iterator.h
 *
 *  Created on: Dec 11, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_ALGROITHMS_REFERENCE_ITERATOR_H_
#define BLACKCAT_TENSORS_ALGROITHMS_REFERENCE_ITERATOR_H_

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


#endif /* REFERENCE_ITERATOR_H_ */
