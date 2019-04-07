/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_STRIDED_VECTOR_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_STRIDED_VECTOR_H_

#include "Array_Base.h"


namespace BC {
namespace exprs {


template<class Parent>
struct Array_Strided_Vector : Array_Base<Array_Strided_Vector<Parent>, 1>, Shape<1> {

    static_assert(Parent::DIMS == 2, "A ROW VIEW MAY ONLY BE CONSTRUCTED FROM A MATRIX");

	using value_type = typename Parent::value_type;
    using allocator_t = typename Parent::allocator_t;
    using system_tag = typename Parent::system_tag;
    static constexpr int  ITERATOR = meta::max(Parent::ITERATOR - 1, 1);
    static constexpr int DIMS = 1;

    value_type* array_slice;

    BCINLINE Array_Strided_Vector(value_type* array_slice_, BC::size_t length, BC::size_t stride)
     : Shape<1>(length, stride),
       array_slice(array_slice_) {}

    BCINLINE const auto& operator [] (int i) const { return array_slice[this->leading_dimension(0) * i]; }
    BCINLINE       auto& operator [] (int i)       { return array_slice[this->leading_dimension(0) * i]; }

    template<class... seq> BCINLINE
    const auto& operator () (int i, seq... indexes) const { return *this[i]; }

    template<class... seq> BCINLINE
    auto& operator () (int i, seq... indexes)        { return *this[i]; }

    BCINLINE const value_type* memptr() const { return array_slice; }
    BCINLINE       value_type* memptr()       { return array_slice; }

};


template<class internal_t>
auto make_row(internal_t internal, BC::size_t  index) {
    return Array_Strided_Vector<internal_t>(&internal.memptr()[index], internal.dimension(0), internal.leading_dimension(0));
}
template<class internal_t>
auto make_diagnol(internal_t internal, BC::size_t  index) {
    BC::size_t  stride = internal.leading_dimension(0) + 1;

    if (index == 0) {
        BC::size_t  length = meta::min(internal.rows(), internal.cols());
        return Array_Strided_Vector<internal_t>(internal.memptr(), length, stride);
    }

    else if (index > 0) {
        BC::size_t  length = meta::min(internal.rows(), internal.cols() - index);
        BC::size_t  location = index * internal.leading_dimension(0);
        return Array_Strided_Vector<internal_t>(&internal.memptr()[location], length, stride);
    }

    else { // (index < 0)  {
        BC::size_t  length = meta::min(internal.rows(), internal.cols() + index);
        return Array_Strided_Vector<internal_t>(&internal.memptr()[std::abs(index)], length, stride);
    }
}


}
}


#endif /* ARRAY_ROW_H_ */
