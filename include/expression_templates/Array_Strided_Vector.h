/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef ARRAY_ROW_H_
#define ARRAY_ROW_H_

#include "Array_Base.h"

namespace BC {
namespace et     {

template<class PARENT>
struct Array_Strided_Vector : Array_Base<Array_Strided_Vector<PARENT>, 1>, Shape<1> {

    using scalar_t = typename PARENT::scalar_t;
    using allocator_t = typename PARENT::allocator_t;
    using system_tag = typename PARENT::system_tag;


    static_assert(PARENT::DIMS() == 2, "A ROW VIEW MAY ONLY BE CONSTRUCTED FROM A MATRIX");

    __BCinline__ static constexpr int ITERATOR() { return MTF::max(PARENT::ITERATOR() - 1, 0); }
    __BCinline__ static constexpr int DIMS() { return MTF::max(PARENT::DIMS() - 1, 0); }

    scalar_t* array_slice;

    __BCinline__ Array_Strided_Vector(const scalar_t* array_slice_, int length, int stride) :
        Shape<1>(length, stride),
        array_slice(const_cast<scalar_t*>(array_slice_)) {}
    __BCinline__ const auto& operator [] (int i) const { return array_slice[this->leading_dimension(0) * i]; }
    __BCinline__        auto& operator [] (int i)        { return array_slice[this->leading_dimension(0) * i]; }

    template<class... seq> __BCinline__
    const auto& operator () (int i, seq... indexes) const { return *this[i]; }

    template<class... seq> __BCinline__
    auto& operator () (int i, seq... indexes)        { return *this[i]; }

    __BCinline__ const scalar_t* memptr() const { return array_slice; }
    __BCinline__       scalar_t* memptr()       { return array_slice; }

};


template<class internal_t>
auto make_row(internal_t internal, int index) {
    return Array_Strided_Vector<internal_t>(&internal.memptr()[index], internal.dimension(0), internal.leading_dimension(0));
}
template<class internal_t>
auto make_diagnol(internal_t internal, int index) {
    int stride = internal.leading_dimension(0) + 1;

    if (index == 0) {
        int length = MTF::min(internal.rows(), internal.cols());
        return Array_Strided_Vector<internal_t>(internal.memptr(), length, stride);
    }

    else if (index > 0) {
        int length = MTF::min(internal.rows(), internal.cols() - index);
        int location = index * internal.leading_dimension(0);
        return Array_Strided_Vector<internal_t>(&internal.memptr()[location], length, stride);
    }
    else { // (index < 0)  {
        int length = MTF::min(internal.rows(), internal.cols() + index);
        return Array_Strided_Vector<internal_t>(&internal.memptr()[std::abs(index)], length, stride);
    }
}

}

}





#endif /* ARRAY_ROW_H_ */
