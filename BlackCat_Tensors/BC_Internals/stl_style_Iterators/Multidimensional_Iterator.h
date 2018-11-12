/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef ITERATOR_H_
#define ITERATOR_H_

#include "Iterator_Base.h"
#include "Coefficientwise_Iterator.h"
namespace BC {
namespace module {
namespace stl {

template<direction direction, class tensor_t>
struct Multidimensional_Iterator :
		IteratorBase<Multidimensional_Iterator<direction, tensor_t>, direction, tensor_t&> {

    using self =Multidimensional_Iterator<direction, tensor_t>;
    using parent = IteratorBase<self, direction, tensor_t>;
    using Iterator = Multidimensional_Iterator<direction, tensor_t>;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = decltype(std::declval<tensor_t&>().slice(0));
    using difference_type = int;
    using reference = value_type;

    __BCinline__ Multidimensional_Iterator(tensor_t& tensor_, int index_=0) :
        parent(tensor_, index_) {}

    using parent::operator=;
    __BCinline__ const value_type operator*() const { return this->tensor[this->index]; }
    __BCinline__ value_type operator*() { return this->tensor[this->index]; }

};


template<class derived_t, typename=std::enable_if_t<derived_t::DIMS() != 1>>
auto forward_iterator_begin(derived_t& derived) {
//    std::cout << " iter begin " << derived.data() << std::endl;
     return Multidimensional_Iterator<direction::forward, derived_t>(derived, 0);
}
template<class derived_t, typename=std::enable_if_t<derived_t::DIMS() != 1>>
auto forward_iterator_end(derived_t& derived) {
     return Multidimensional_Iterator<direction::forward, derived_t>(derived, derived.outer_dimension());
}

template<class derived_t, typename=std::enable_if_t<derived_t::DIMS() != 1>>
auto reverse_iterator_begin(derived_t& derived) {
     return Multidimensional_Iterator<direction::reverse, derived_t>(derived, derived.outer_dimension()-1);
}

template<class derived_t, typename=std::enable_if_t<derived_t::DIMS() != 1>>
auto reverse_iterator_end(derived_t& derived) {
     return Multidimensional_Iterator<direction::reverse, derived_t>(derived, -1);
}

template<class derived_t>
std::enable_if_t<derived_t::DIMS() == 1, decltype(forward_cwise_iterator_begin(std::declval<derived_t&>()))>
forward_iterator_begin(derived_t& derived) {
    return forward_cwise_iterator_begin(derived);
}

template<class derived_t, class = std::enable_if_t<derived_t::DIMS() == 1>>
std::enable_if_t<derived_t::DIMS() == 1, decltype(forward_cwise_iterator_end(std::declval<derived_t&>()))>
forward_iterator_end(derived_t& derived) {
    return forward_cwise_iterator_end(derived);
}

template<class derived_t, class = std::enable_if_t<derived_t::DIMS() == 1>>
std::enable_if_t<derived_t::DIMS() == 1, decltype(reverse_cwise_iterator_begin(std::declval<derived_t&>()))>
reverse_iterator_begin(derived_t& derived) {
    return reverse_cwise_iterator_begin(derived);
}

template<class derived_t, class = std::enable_if_t<derived_t::DIMS() == 1>>
std::enable_if_t<derived_t::DIMS() == 1, decltype(reverse_cwise_iterator_begin(std::declval<derived_t&>()))>
reverse_iterator_end(derived_t& derived) {
    return reverse_cwise_iterator_begin(derived);
}


}
}
}


#endif /* ITERATOR_H_ */
