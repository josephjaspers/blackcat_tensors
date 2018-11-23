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
struct Multidimensional_Iterator {

    using self =Multidimensional_Iterator<direction, tensor_t>;
    using Iterator = Multidimensional_Iterator<direction, tensor_t>;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = decltype(std::declval<tensor_t&>().slice(0));
    using difference_type = int;
    using reference = value_type;

    tensor_t& tensor;
    int index;

    __BCinline__ Multidimensional_Iterator(tensor_t& tensor_, int index_=0) :
			tensor(tensor_), index(index_) {}

#define BC_ND_Iter_Compare(sign, rev)                          \
	__BCinline__											\
    bool operator sign (const Iterator& iter) {             \
        if (direction == direction::forward)                \
            return index sign iter.index;                   \
        else                                                \
            return index rev iter.index;                    \
    }                                                       \
    __BCinline__ 											\
    bool operator sign (int p_index) {                      \
        if (direction == direction::forward)                \
            return index sign p_index;                      \
        else                                                \
            return index rev  p_index;                      \
    }

    BC_ND_Iter_Compare(<, >)
    BC_ND_Iter_Compare(>, <)
    BC_ND_Iter_Compare(<=, >=)
    BC_ND_Iter_Compare(>=, <=)

    __BCinline__ operator int () const { return index; }

    __BCinline__ bool operator == (const Iterator& iter) {
        return index == iter.index;
    }
    __BCinline__ bool operator != (const Iterator& iter) {
        return index != iter.index;
    }

    __BCinline__ Iterator& operator =  (int index_) { this->index = index_;  return *this; }

    __BCinline__ Iterator& operator ++ () { index+=direction; return *this; }
    __BCinline__ Iterator& operator -- () { index-=direction; return *this; }

	__BCinline__ Iterator operator ++(int) { return Iterator(tensor, index++); }
	__BCinline__ Iterator operator --(int) { return Iterator(tensor, index--); }

    __BCinline__ Iterator& operator += (int dist)    { index += dist*direction; return *this; }
    __BCinline__ Iterator& operator -= (int dist)    { index -= dist*direction; return *this; }

    __BCinline__ Iterator operator + (int dist) const { return Iterator(tensor, index + dist*direction); }
    __BCinline__ Iterator operator - (int dist) const { return Iterator(tensor, index - dist*direction); }

    __BCinline__ Iterator& operator += (const Iterator& dist) { index += dist.index*direction; return *this; }
    __BCinline__ Iterator& operator -= (const Iterator& dist) { index -= dist.index*direction; return *this; }
    __BCinline__ Iterator operator + (const Iterator& dist) const { return Iterator(tensor, index + dist.index*direction); }
    __BCinline__ Iterator operator - (const Iterator& dist) const { return Iterator(tensor, index - dist.index*direction); }

    const value_type operator*() const { return this->tensor[this->index]; }
    value_type operator*() { return this->tensor[this->index]; }

};


template<class derived_t, typename=std::enable_if_t<derived_t::DIMS() != 1>>
auto forward_iterator_begin(derived_t& derived) {
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
std::enable_if_t<derived_t::DIMS() == 1, decltype(forward_cwise_iterator_begin(std::declval<derived_t&>().internal()))>
forward_iterator_begin(derived_t& derived) {
    return forward_cwise_iterator_begin(derived.internal());
}

template<class derived_t, class = std::enable_if_t<derived_t::DIMS() == 1>>
std::enable_if_t<derived_t::DIMS() == 1, decltype(forward_cwise_iterator_end(std::declval<derived_t&>().internal()))>
forward_iterator_end(derived_t& derived) {
    return forward_cwise_iterator_end(derived.internal());
}

template<class derived_t, class = std::enable_if_t<derived_t::DIMS() == 1>>
std::enable_if_t<derived_t::DIMS() == 1, decltype(reverse_cwise_iterator_begin(std::declval<derived_t&>().internal()))>
reverse_iterator_begin(derived_t& derived) {
    return reverse_cwise_iterator_begin(derived.internal());
}

template<class derived_t, class = std::enable_if_t<derived_t::DIMS() == 1>>
std::enable_if_t<derived_t::DIMS() == 1, decltype(reverse_cwise_iterator_begin(std::declval<derived_t&>().internal()))>
reverse_iterator_end(derived_t& derived) {
    return reverse_cwise_iterator_begin(derived.internal());
}


}
}
}


#endif /* ITERATOR_H_ */
