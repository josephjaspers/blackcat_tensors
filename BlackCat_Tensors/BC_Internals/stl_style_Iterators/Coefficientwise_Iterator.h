/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef ELEMENTWISE_Coefficientwise_Iterator_H_
#define ELEMENTWISE_Coefficientwise_Iterator_H_

namespace BC {
namespace module {
namespace stl {

template<direction direction, class tensor_t>
struct Coefficientwise_Iterator {

	using Coefficientwise_Iterator_category = std::random_access_iterator_tag;
	using value_type = decltype(std::declval<tensor_t>()(0)); //operator() == scalar access operator
	using difference_type = int;
	using pointer = tensor_t&;
	using reference = value_type;

private:
	bool same_ptr(const Coefficientwise_Iterator& iter) const { return tensor.data() == iter.tensor.data(); }
	bool same_index(const Coefficientwise_Iterator& iter) const { return index == iter.index; }
public:

	static_assert(tensor_t::DIMS() > 0, "Iterator not defined for scalar_types");
	static_assert(tensor_t::ITERATOR() == 0 || tensor_t::ITERATOR() == 1,
			"Elementwise-Iterator only available to continuous tensors");


	tensor_t& tensor;
	mutable int index= 0;

	Coefficientwise_Iterator(tensor_t& tensor_, int index_=0)
	: tensor(tensor_), index(index_) {
	}

	bool operator == (const Coefficientwise_Iterator& iter) {
		return same_index(iter) && same_ptr(iter);
	}
	bool operator != (const Coefficientwise_Iterator& iter) {
		return !((*this)==iter);
	}

#define BC_Iter_Cwise_Compare(sign, rev) 							 \
	bool operator sign (const Coefficientwise_Iterator& iter) {  \
		if (direction == direction::forward)				 \
			return index sign iter.index && same_ptr(iter);  \
		else 												 \
			return index rev iter.index && same_ptr(iter);	 \
	}

	BC_Iter_Cwise_Compare(<, >)
	BC_Iter_Cwise_Compare(>, <)
	BC_Iter_Cwise_Compare(<=, >=)
	BC_Iter_Cwise_Compare(>=, <=)

	Coefficientwise_Iterator& operator =  (int index_) { this->index = index_;  return *this; }

	Coefficientwise_Iterator& operator ++ () { index+=direction; return *this; }
	Coefficientwise_Iterator& operator -- () { index-=direction; return *this; }

	Coefficientwise_Iterator operator ++ (int) { Coefficientwise_Iterator tmp = *this; index+=direction; return tmp; }
	Coefficientwise_Iterator operator -- (int) { Coefficientwise_Iterator tmp = *this; index-=direction; return tmp; }

	Coefficientwise_Iterator& operator += (int dist)    { index += dist*direction; return *this; }
	Coefficientwise_Iterator& operator -= (int dist)    { index -= dist*direction; return *this; }

	Coefficientwise_Iterator operator + (int dist) const { return Coefficientwise_Iterator(tensor, index + dist*direction); }
	Coefficientwise_Iterator operator - (int dist) const { return Coefficientwise_Iterator(tensor, index - dist*direction); }

	Coefficientwise_Iterator& operator += (const Coefficientwise_Iterator& dist) const { index += dist.index*direction; return *this; }
	Coefficientwise_Iterator& operator -= (const Coefficientwise_Iterator& dist) const { index -= dist.index*direction; return *this; }

	Coefficientwise_Iterator operator + (const Coefficientwise_Iterator& dist)  const { return Coefficientwise_Iterator(tensor, index + dist.index*direction); }
	Coefficientwise_Iterator operator - (const Coefficientwise_Iterator& dist)  const { return Coefficientwise_Iterator(tensor, index - dist.index*direction); }

	value_type operator*() const { return tensor(index); }
};

template<class tensor_t>
auto forward_cwise_iterator_begin(tensor_t& tensor) {
	return Coefficientwise_Iterator<direction::forward, tensor_t>(tensor, 0);
}
template<class tensor_t>
auto forward_cwise_iterator_end(tensor_t& tensor) {
	return Coefficientwise_Iterator<direction::forward, tensor_t>(tensor, tensor.size());
}
template<class tensor_t>
auto reverse_cwise_iterator_begin(tensor_t& tensor) {
	return Coefficientwise_Iterator<direction::reverse, tensor_t>(tensor, tensor.size()-1);
}
template<class tensor_t>
auto reverse_cwise_iterator_end(tensor_t& tensor) {
	return Coefficientwise_Iterator<direction::reverse, tensor_t>(tensor, -1);
}
}
}
}



#endif /* ELEMENTWISE_Coefficientwise_Iterator_H_ */
