/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef ELEMENTWISE_Coefficientwise_Iterator_H_
#define ELEMENTWISE_Coefficientwise_Iterator_H_

#include "Iterator_Base.h"

namespace BC {
namespace module {
namespace stl {

template<direction direction, class tensor_t>
struct Coefficientwise_Iterator  : IteratorBase<Coefficientwise_Iterator<direction, tensor_t>, direction, tensor_t>{

	using self = Coefficientwise_Iterator<direction, tensor_t>;
	using parent = IteratorBase<self, direction, tensor_t>;
	using iterator_category = std::random_access_iterator_tag;
	using value_type = std::decay_t<decltype(std::declval<tensor_t&>().memptr()[0])>;
	using difference_type = int;
	using pointer = std::decay_t<value_type>*;
	using reference = value_type&;

	parent::operator=;

//	static_assert(tensor_t::DIMS() > 0, "Iterator not defined for scalar_types");
	static_assert(tensor_t::ITERATOR() == 0 || tensor_t::ITERATOR() == 1,
			"Elementwise-Iterator only available to continuous tensors");

	Coefficientwise_Iterator(tensor_t& tensor_, int index_=0)
	: parent(tensor_, index_) {
	}

	Coefficientwise_Iterator& operator =(const Coefficientwise_Iterator& iter) {
		this->index = iter.index;
		return *this;
	}

	reference operator*() const { return this->tensor.data()[this->index]; }
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
