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

template<direction direction, class tensor_t_, class enabler=void>
struct Coefficientwise_Iterator {

    using Iterator = Coefficientwise_Iterator<direction, tensor_t_>;
    using tensor_t = tensor_t_;

    static constexpr bool ref_value_type
    	= std::is_reference<decltype(std::declval<tensor_t>().internal()[0])>::value;

    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename tensor_t::value_type;
    using difference_type = int;
    using reference = decltype(std::declval<tensor_t>()[0]);
    using pointer   = std::conditional_t<std::is_same<reference, value_type>::value,
    										decltype(std::declval<tensor_t>().internal()), value_type*>;


    tensor_t tensor;
    BC::size_t  index;

    BCINLINE Coefficientwise_Iterator(tensor_t tensor_, BC::size_t  index_=0) :
	tensor(tensor_), index(index_) {}

    BCINLINE Coefficientwise_Iterator& operator =(const Coefficientwise_Iterator& iter) {
        this->index = iter.index;
        return *this;
    }

#define BC_Iter_Compare(sign, rev)                          \
	BCINLINE				            \
    bool operator sign (const Iterator& iter) {             \
        if (direction == direction::forward)                \
            return index sign iter.index;                   \
        else                                                \
            return index rev iter.index;                    \
    }                                                       \
    BCINLINE 					    \
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

#undef BC_Iter_Compare

    BCINLINE operator BC::size_t  () const { return index; }

    BCINLINE bool operator == (const Iterator& iter) {
        return index == iter.index;
    }
    BCINLINE bool operator != (const Iterator& iter) {
        return index != iter.index;
    }

    BCINLINE Iterator& operator =  (int index_) { this->index = index_;  return *this; }

    BCINLINE Iterator& operator ++ ()    { index += direction; return *this; }
    BCINLINE Iterator& operator -- ()    { index -= direction; return *this; }

    BCINLINE Iterator  operator ++ (int) { return Iterator(tensor, index++); }
    BCINLINE Iterator  operator -- (int) { return Iterator(tensor, index--); }


    BCINLINE Iterator& operator += (int dist)       { index += dist*direction; return *this; }
    BCINLINE Iterator& operator -= (int dist)       { index -= dist*direction; return *this; }

    BCINLINE Iterator  operator +  (int dist) const { return Iterator(tensor, index + dist*direction); }
    BCINLINE Iterator  operator -  (int dist) const { return Iterator(tensor, index - dist*direction); }


    BCINLINE Iterator& operator += (const Iterator& dist)       { index += dist.index * direction; return *this; }
    BCINLINE Iterator& operator -= (const Iterator& dist)       { index -= dist.index * direction; return *this; }
    
    BCINLINE Iterator  operator +  (const Iterator& dist) const { return Iterator(tensor, index + dist.index*direction); }
    BCINLINE Iterator  operator -  (const Iterator& dist) const { return Iterator(tensor, index - dist.index*direction); }


    BCINLINE auto operator*() const -> decltype(this->tensor[this->index]) { return this->tensor[this->index]; }
    BCINLINE auto operator*() 	 	-> decltype(this->tensor[this->index]) { return this->tensor[this->index]; }
    
    BCINLINE auto operator [] (int i) const -> decltype(this->tensor[i]) { return this->tensor[i]; }
    BCINLINE auto operator [] (int i)       -> decltype(this->tensor[i]) { return this->tensor[i]; }
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
