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

    static_assert(tensor_t::ITERATOR() == 0 || tensor_t::ITERATOR() == 1, "Elementwise-Iterator only available to continuous tensors");

    using self = Coefficientwise_Iterator<direction, tensor_t>;
    using parent = IteratorBase<self, direction, tensor_t>;

    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename tensor_t::scalar_t;
    using difference_type = int;
    using pointer =  value_type*;
    using reference = value_type&;

    parent::operator=;

    __BCinline__ Coefficientwise_Iterator(tensor_t tensor_, int index_=0)
    	: parent(tensor_, index_) {}

    __BCinline__ Coefficientwise_Iterator& operator =(const Coefficientwise_Iterator& iter) {
        this->index = iter.index;
        return *this;
    }

    __BCinline__ auto& operator*() const { return this->tensor[this->index]; }
    __BCinline__ auto& operator*() { return this->tensor[this->index]; }
    __BCinline__ auto& operator [] (int i) const { return this->tensor[i]; }
    __BCinline__ auto& operator [] (int i)       { return this->tensor[i]; }
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
