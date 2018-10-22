/*
 * Iterator.h
 *
 *  Created on: Oct 18, 2018
 *      Author: joseph
 */

#ifndef ITERATOR_H_
#define ITERATOR_H_

#include "STL_Iterator_Common.h"

namespace BC {
namespace module {
namespace stl {

template<direction direction, class tensor_t>
struct Iterator {

	using iterator_category = std::random_access_iterator_tag;
	using value_type = decltype(std::declval<tensor_t>().slice(0));
	using difference_type = int;
	using pointer = tensor_t&;
	using reference = value_type;

private:
	bool same_ptr(const Iterator& iter) const { return tensor.data() == iter.tensor.data(); }
	bool same_index(const Iterator& iter) const { return index == iter.index; }
public:

	tensor_t& tensor;
	mutable int index= 0;

	Iterator(tensor_t& tensor_, int index_=0)
	: tensor(tensor_), index(index_) {
	}

	bool operator == (const Iterator& iter) {
		return same_index(iter) && same_ptr(iter);
	}
	bool operator != (const Iterator& iter) {
		return !((*this)==iter);
	}

#define BC_Iter_Compare(sign, rev) 							\
	bool operator sign (const Iterator& iter) {    	 	     \
		if (direction == direction::forward)						 \
			return index sign iter.index && same_ptr(iter);  \
		else 												 \
			return index rev iter.index && same_ptr(iter);		\
	}

	BC_Iter_Compare(<, >)
	BC_Iter_Compare(>=, <=)

	Iterator& operator =  (int index_) { this->index = index_;  return *this; }

	Iterator& operator ++ () { index+=direction; return *this; }
	Iterator& operator -- () { index-=direction; return *this; }

	Iterator operator ++ (int) { Iterator tmp = *this; index+=direction; return tmp; }
	Iterator operator -- (int) { Iterator tmp = *this; index-=direction; return tmp; }

	Iterator& operator += (int dist)    { index += dist*direction; return *this; }
	Iterator& operator -= (int dist)    { index -= dist*direction; return *this; }

	Iterator operator + (int dist) const { return Iterator(tensor, index + dist*direction); }
	Iterator operator - (int dist) const { return Iterator(tensor, index - dist*direction); }

	Iterator& operator += (const Iterator& dist) const { index += dist.index*direction; return *this; }
	Iterator& operator -= (const Iterator& dist) const { index -= dist.index*direction; return *this; }

	Iterator operator + (const Iterator& dist)  const { return Iterator(tensor, index + dist.index*direction); }
	Iterator operator - (const Iterator& dist)  const { return Iterator(tensor, index - dist.index*direction); }

	value_type operator*() const { return tensor[index]; }
};


template<class derived_t>
auto forward_iterator_begin(derived_t& derived) {
//	std::cout << " iter begin " << derived.data() << std::endl;
	 return Iterator<direction::forward, derived_t>(derived, 0);
}
template<class derived_t>
auto forward_iterator_end(derived_t& derived) {
	 return Iterator<direction::forward, derived_t>(derived, derived.outer_dimension());
}

template<class derived_t>
auto reverse_iterator_begin(derived_t& derived) {
	 return Iterator<direction::reverse, derived_t>(derived, derived.outer_dimension()-1);
}

template<class derived_t>
auto reverse_iterator_end(derived_t& derived) {
	 return Iterator<direction::reverse, derived_t>(derived, -1);
}

}
}
}


#endif /* ITERATOR_H_ */
