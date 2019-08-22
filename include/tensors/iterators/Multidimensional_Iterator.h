/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef tensor_iterator_dimension_H_
#define tensor_iterator_dimension_H_

#include "Common.h"

namespace BC {
namespace tensors {
namespace iterators {

template<direction direction, class Tensor>
struct Multidimensional_Iterator {

    using self =Multidimensional_Iterator<direction, Tensor>;
    using Iterator = Multidimensional_Iterator<direction, Tensor>;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = decltype(std::declval<Tensor>().slice(0));
    using difference_type = int;
    using reference = value_type;

    Tensor& tensor;
    BC::size_t  index;

    BCINLINE Multidimensional_Iterator(Tensor& tensor_, BC::size_t  index_=0) :
			tensor(tensor_), index(index_) {}

#define BC_ND_Iter_Compare(sign, rev)                          \
	BCINLINE											\
    bool operator sign (const Iterator& iter) {             \
        if (direction == direction::forward)                \
            return index sign iter.index;                   \
        else                                                \
            return index rev iter.index;                    \
    }                                                       \
    BCINLINE 											\
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
#undef BC_ND_Iter_Compare

    BCINLINE operator BC::size_t  () const { return index; }

    BCINLINE bool operator == (const Iterator& iter) {
        return index == iter.index;
    }
    BCINLINE bool operator != (const Iterator& iter) {
        return index != iter.index;
    }

    BCINLINE Iterator& operator =  (int index_) { this->index = index_;  return *this; }

    BCINLINE Iterator& operator ++ () { return *this += direction; }
    BCINLINE Iterator& operator -- () { return *this += direction; }

	BCINLINE Iterator operator ++(int) { return Iterator(tensor, index++); }
	BCINLINE Iterator operator --(int) { return Iterator(tensor, index--); }

    BCINLINE Iterator& operator += (int dist)    { index += dist*direction; return *this; }
    BCINLINE Iterator& operator -= (int dist)    { index -= dist*direction; return *this; }

    BCINLINE Iterator operator + (int dist) const { return Iterator(tensor, index+dist*direction); }
    BCINLINE Iterator operator - (int dist) const { return Iterator(tensor, index-dist*direction); }

    BCINLINE Iterator& operator += (const Iterator& dist) { return *this += dist.index; }
    BCINLINE Iterator& operator -= (const Iterator& dist) { return *this -= dist.index; }
    BCINLINE Iterator operator + (const Iterator& dist) const { return Iterator(tensor, *this+dist.index); }
    BCINLINE Iterator operator - (const Iterator& dist) const { return Iterator(tensor, *this-dist.index); }

    const value_type operator*() const { return this->tensor[this->index]; }
    value_type operator*() { return this->tensor[this->index]; }

};


template<class derived_t>//, typename=std::enable_if_t<derived_t::tensor_dimension != 1>>
auto forward_iterator_begin(derived_t& derived) {
     return Multidimensional_Iterator<direction::forward, derived_t>(derived, 0);
}
template<class derived_t>//, typename=std::enable_if_t<derived_t::tensor_dimension != 1>>
auto forward_iterator_end(derived_t& derived) {
     return Multidimensional_Iterator<direction::forward, derived_t>(derived, derived.outer_dimension());
}

template<class derived_t>//, typename=std::enable_if_t<derived_t::tensor_dimension != 1>>
auto reverse_iterator_begin(derived_t& derived) {
     return Multidimensional_Iterator<direction::reverse, derived_t>(derived, derived.outer_dimension()-1);
}

template<class derived_t>//, typename=std::enable_if_t<derived_t::tensor_dimension != 1>>
auto reverse_iterator_end(derived_t& derived) {
     return Multidimensional_Iterator<direction::reverse, derived_t>(derived, -1);
}

}
}
}

#endif /* tensor_iterator_dimension_H_ */
