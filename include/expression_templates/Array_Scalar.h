/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef TENSOR_SCALAR_H_
#define TENSOR_SCALAR_H_

#include "Array_Base.h"

namespace BC {
namespace et     {
/*
 * Represents a single_scalar value from a tensor
 */

template<class Parent>
struct Array_Scalar : Array_Base<Array_Scalar<Parent>, 0>, Shape<0> {

    using value_type = typename Parent::value_type;
    using allocator_t = typename Parent::allocator_t;
    using system_tag = typename Parent::system_tag;

    static constexpr int ITERATOR = 0;
    static constexpr int DIMS = 0;

    value_type* array;

    __BCinline__ Array_Scalar(Parent parent_, BC::size_t  index)
    : array(&(parent_.memptr()[index])) {}

    __BCinline__ const auto& operator [] (int index) const { return array[0]; }
    __BCinline__       auto& operator [] (int index)       { return array[0]; }

    template<class... integers> __BCinline__
    auto& operator ()(integers ... ints) {
        return array[0];
    }
    template<class... integers> __BCinline__
    const auto& operator ()(integers ... ints) const {
        return array[0];
    }

    __BCinline__ const value_type* memptr() const { return array; }
    __BCinline__       value_type* memptr()       { return array; }
};

template<class internal_t>
auto make_scalar(internal_t internal, BC::size_t  i) {
    return Array_Scalar<internal_t>(internal, i);
}
}
}



#endif /* TENSOR_SLICE_CU_ */
