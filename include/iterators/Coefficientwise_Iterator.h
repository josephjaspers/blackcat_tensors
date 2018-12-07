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

template<direction direction, class tensor_t_>
struct Coefficientwise_Iterator {

    using Iterator = Coefficientwise_Iterator<direction, tensor_t_>;
    using tensor_t = tensor_t_;

    static constexpr bool ref_value_type
    	= std::is_reference<decltype(std::declval<tensor_t>().internal()[0])>::value;

    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename tensor_t::scalar_t;
    using difference_type = int;
    using pointer =  value_type*;
    using reference = value_type&;


    tensor_t tensor;
    int index;

    __BCinline__ Coefficientwise_Iterator(tensor_t tensor_, int index_=0) :
	tensor(tensor_), index(index_) {}

    __BCinline__ Coefficientwise_Iterator& operator =(const Coefficientwise_Iterator& iter) {
        this->index = iter.index;
        return *this;
    }

#define BC_Iter_Compare(sign, rev)                          \
	__BCinline__				            \
    bool operator sign (const Iterator& iter) {             \
        if (direction == direction::forward)                \
            return index sign iter.index;                   \
        else                                                \
            return index rev iter.index;                    \
    }                                                       \
    __BCinline__ 					    \
    bool operator sign (int p_index) {                      \
        if (direction == direction::forward)                \
            return index sign p_index;                      \
        else                                                \
            return index rev  p_index;                      \
    }

    BC_Iter_Compare(<, >)
    BC_Iter_Compare(>, <)
    BC_Iter_Compare(<=, >=)
    BC_Iter_Compare(>=, <=)

    __BCinline__ operator int () const { return index; }

    __BCinline__ bool operator == (const Iterator& iter) {
        return index == iter.index;
    }
    __BCinline__ bool operator != (const Iterator& iter) {
        return index != iter.index;
    }

    __BCinline__ Iterator& operator =  (int index_) { this->index = index_;  return *this; }

    __BCinline__ Iterator& operator ++ ()    { index += direction; return *this; }
    __BCinline__ Iterator& operator -- ()    { index -= direction; return *this; }

    __BCinline__ Iterator  operator ++ (int) { return Iterator(tensor, index++); }
    __BCinline__ Iterator  operator -- (int) { return Iterator(tensor, index--); }


    __BCinline__ Iterator& operator += (int dist)       { index += dist*direction; return *this; }
    __BCinline__ Iterator& operator -= (int dist)       { index -= dist*direction; return *this; }

    __BCinline__ Iterator  operator +  (int dist) const { return Iterator(tensor, index + dist*direction); }
    __BCinline__ Iterator  operator -  (int dist) const { return Iterator(tensor, index - dist*direction); }


    __BCinline__ Iterator& operator += (const Iterator& dist)       { index += dist.index * direction; return *this; }
    __BCinline__ Iterator& operator -= (const Iterator& dist)       { index -= dist.index * direction; return *this; }
    
    __BCinline__ Iterator  operator +  (const Iterator& dist) const { return Iterator(tensor, index + dist.index*direction); }
    __BCinline__ Iterator  operator -  (const Iterator& dist) const { return Iterator(tensor, index - dist.index*direction); }


    __BCinline__ auto& operator*() const { return this->tensor[this->index]; }
    __BCinline__ auto& operator*() 	 { return this->tensor[this->index]; }
    
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
