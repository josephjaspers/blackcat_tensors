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

template<class PARENT>
struct Array_Scalar : Array_Base<Array_Scalar<PARENT>, 0>, Shape<0> {

    using scalar_t = typename PARENT::scalar_t;
    using allocator_t = typename PARENT::allocator_t;
    using system_tag = typename PARENT::system_tag;

    __BCinline__ static constexpr int ITERATOR() { return 0; }
    __BCinline__ static constexpr int DIMS()      { return 0; }

    scalar_t* array;

    __BCinline__ Array_Scalar(PARENT parent_, int index) : array(const_cast<scalar_t*>(&(parent_.memptr()[index]))) {}

    __BCinline__ const auto& operator [] (int index) const { return array[0]; }
    __BCinline__        auto& operator [] (int index)        { return array[0]; }

    template<class... integers> __BCinline__
    auto& operator ()(integers ... ints) {
        return array[0];
    }
    template<class... integers> __BCinline__
    const auto& operator ()(integers ... ints) const {
        return array[0];
    }

    __BCinline__ const scalar_t* memptr() const { return array; }
    __BCinline__       scalar_t* memptr()          { return array; }
};

template<class internal_t>
auto make_scalar(internal_t internal, int i) {
    return Array_Scalar<internal_t>(internal, i);
}
}
}



#endif /* TENSOR_SLICE_CU_ */
