/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef ELEMENTWISE_Coefficientwise_Iterator_H_
#define ELEMENTWISE_Coefficientwise_Iterator_H_

#include "Common.h"

namespace BC {
namespace tensors {
namespace iterators {

template<direction Direction, class Tensor, class enabler=void>
struct Coefficientwise_Iterator {

    using Iterator = Coefficientwise_Iterator<Direction, Tensor>;

    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename Tensor::value_type;
    using difference_type = BC::size_t;
    using reference = value_type&;
    using pointer   = value_type*;


    Tensor tensor;
    BC::size_t  index;

    BCINLINE Coefficientwise_Iterator(Tensor tensor_, BC::size_t  index_=0) :
	tensor(tensor_), index(index_) {}

    BCINLINE Coefficientwise_Iterator& operator =(const Coefficientwise_Iterator& iter) {
        this->index = iter.index;
        return *this;
    }

#define BC_Iter_Compare(sign, rev)                          \
	BCINLINE				            \
    bool operator sign (const Iterator& iter) {             \
        if (Direction == direction::forward)                \
            return index sign iter.index;                   \
        else                                                \
            return index rev iter.index;                    \
    }                                                       \
    BCINLINE 					    \
    bool operator sign (int p_index) {                      \
        if (Direction == direction::forward)                \
            return index sign p_index;                      \
        else                                                \
            return index rev  p_index;                      \
    }

    BC_Iter_Compare(<, >)
    BC_Iter_Compare(>, <)
    BC_Iter_Compare(<=, >=)
    BC_Iter_Compare(>=, <=)

#undef BC_Iter_Compare

    BCINLINE operator BC::size_t  () const { return index; }

    BCINLINE bool operator == (const Iterator& iter) {
        return index == iter.index;
    }
    BCINLINE bool operator != (const Iterator& iter) {
        return index != iter.index;
    }

    BCINLINE Iterator& operator =  (int index_) { this->index = index_;  return *this; }

    BCINLINE Iterator& operator ++ ()    { index += Direction; return *this; }
    BCINLINE Iterator& operator -- ()    { index -= Direction; return *this; }

    BCINLINE Iterator  operator ++ (int) { return Iterator(tensor, index++); }
    BCINLINE Iterator  operator -- (int) { return Iterator(tensor, index--); }


    BCINLINE Iterator& operator += (int dist)       { index += dist*Direction; return *this; }
    BCINLINE Iterator& operator -= (int dist)       { index -= dist*Direction; return *this; }

    BCINLINE Iterator  operator +  (int dist) const { return Iterator(tensor, index + dist*Direction); }
    BCINLINE Iterator  operator -  (int dist) const { return Iterator(tensor, index - dist*Direction); }


    BCINLINE Iterator& operator += (const Iterator& dist)       { index += dist.index * Direction; return *this; }
    BCINLINE Iterator& operator -= (const Iterator& dist)       { index -= dist.index * Direction; return *this; }
    
    BCINLINE Iterator  operator +  (const Iterator& dist) const { return Iterator(tensor, index + dist.index*Direction); }
    BCINLINE Iterator  operator -  (const Iterator& dist) const { return Iterator(tensor, index - dist.index*Direction); }


    BCINLINE auto operator*() const -> decltype(this->tensor[this->index]) { return this->tensor[this->index]; }
    BCINLINE auto operator*() 	 	-> decltype(this->tensor[this->index]) { return this->tensor[this->index]; }
    
    BCINLINE auto operator [] (int i) const -> decltype(this->tensor[i]) { return this->tensor[i]; }
    BCINLINE auto operator [] (int i)       -> decltype(this->tensor[i]) { return this->tensor[i]; }
};

template<class Tensor>
auto forward_cwise_iterator_begin(Tensor& tensor) {
    return Coefficientwise_Iterator<direction::forward, Tensor>(tensor, 0);
}
template<class Tensor>
auto forward_cwise_iterator_end(Tensor& tensor) {
    return Coefficientwise_Iterator<direction::forward, Tensor>(tensor, tensor.size());
}
template<class Tensor>
auto reverse_cwise_iterator_begin(Tensor& tensor) {
    return Coefficientwise_Iterator<direction::reverse, Tensor>(tensor, tensor.size()-1);
}
template<class Tensor>
auto reverse_cwise_iterator_end(Tensor& tensor) {
    return Coefficientwise_Iterator<direction::reverse, Tensor>(tensor, -1);
}


}
}
}

#endif /* ELEMENTWISE_Coefficientwise_Iterator_H_ */
