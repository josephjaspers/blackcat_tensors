/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef tensor_iterator_dim_H_
#define tensor_iterator_dim_H_

#include "common.h"

namespace bc {
namespace tensors {
namespace iterators {

template<direction direction, class Tensor>
struct Multidimensional_Iterator {

    using self = Multidimensional_Iterator<direction, Tensor>;
    using Iterator = Multidimensional_Iterator<direction, Tensor>;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = decltype(std::declval<Tensor>().slice(0));
    using difference_type = int;
    using reference = value_type;

    Tensor& tensor;
    bc::size_t  index;

    BCINLINE Multidimensional_Iterator(Tensor& tensor_, bc::size_t  index_=0) :
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

    BCINLINE operator bc::size_t  () const { return index; }

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


template<class Tensor>
auto iter_begin(Tensor& derived) {
	return Multidimensional_Iterator<direction::forward, Tensor>(derived, 0);
}

template<class Tensor>
auto iter_end(Tensor& derived) {
	return Multidimensional_Iterator<direction::forward, Tensor>(derived, derived.outer_dim());
}

template<class Tensor>
auto iter_rbegin(Tensor& derived) {
	return Multidimensional_Iterator<direction::reverse, Tensor>(derived, derived.outer_dim()-1);
}

template<class Tensor>
auto iter_rend(Tensor& derived) {
	return Multidimensional_Iterator<direction::reverse, Tensor>(derived, -1);
}

}
}
}

#endif /* tensor_iterator_dim_H_ */
