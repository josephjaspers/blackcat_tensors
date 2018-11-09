/*
 * IteratorBase_Base.h
 *
 *  Created on: Oct 28, 2018
 *      Author: joseph
 */

#ifndef ITERATOR_BASE_H_
#define ITERATOR_BASE_H_

#include "STL_Iterator_Common.h"

namespace BC {
namespace module {
namespace stl {

template<class derived, direction direction, class tensor_t_>
struct IteratorBase {

    using Iterator = derived;
    using iterator_category = std::random_access_iterator_tag;
    using tensor_t = std::decay_t<tensor_t_>;
    operator const derived&() const { return static_cast<derived&>(*this); }
    operator derived&() { return  static_cast<derived&>(*this); }

public:

    tensor_t& tensor;
    mutable int index= 0;

    IteratorBase(tensor_t& tensor_, int index_=0)
    : tensor(tensor_), index(index_) {
    }

#define BC_Iter_Compare(sign, rev)                          \
    bool operator sign (const Iterator& iter) {             \
        if (direction == direction::forward)                \
            return index sign iter.index;                   \
        else                                                \
            return index rev iter.index;                    \
    }                                                       \
    bool operator sign (int p_index) {                         \
        if (direction == direction::forward)                \
            return index sign p_index;                      \
        else                                                \
            return index rev  p_index;                      \
    }

    BC_Iter_Compare(<, >)
    BC_Iter_Compare(>, <)
    BC_Iter_Compare(<=, >=)
    BC_Iter_Compare(>=, <=)

    operator int () const { return index; }

    bool operator == (const Iterator& iter) {
        return index == iter.index;
    }
    bool operator != (const Iterator& iter) {
        return index != iter.index;
    }

    Iterator& operator =  (int index_) { this->index = index_;  return *this; }

    Iterator& operator ++ () { index+=direction; return *this; }
    Iterator& operator -- () { index-=direction; return *this; }

    Iterator operator ++ (int) { Iterator tmp = *this; index+=direction; return tmp; }
    Iterator operator -- (int) { Iterator tmp = *this; index-=direction; return tmp; }

    Iterator& operator += (int dist)    { index += dist*direction; return *this; }
    Iterator& operator -= (int dist)    { index -= dist*direction; return *this; }

    Iterator operator + (int dist) const { return Iterator(tensor, index + dist*direction); }
    Iterator operator - (int dist) const { return Iterator(tensor, index - dist*direction); }

    Iterator& operator += (const Iterator& dist) const { index += dist.index*direction; return *this; }
    Iterator& operator -= (const Iterator& dist) const { index -= dist.index*direction; return *this; }

    Iterator operator + (const Iterator& dist)  const { return Iterator(tensor, index + dist.index*direction); }
    Iterator operator - (const Iterator& dist)  const { return Iterator(tensor, index - dist.index*direction); }

};

}
}
}



#endif /* ITERATOR_BASE_H_ */
