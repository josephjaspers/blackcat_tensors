/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef TENSOR_RESHAPE_H_
#define TENSOR_RESHAPE_H_

#include "Array_Base.h"

namespace BC {
namespace et {
/*
 * Accepts an a tensor_core type wrapped in the new_tensor
*/

template<int dimension>
struct Array_Reshape {

    template<class PARENT>
    struct implementation : Array_Base<implementation<PARENT>, dimension>, Shape<dimension> {

    using scalar_t = typename PARENT::scalar_t;
    using allocator_t = typename PARENT::allocator_t;
    using system_tag = typename PARENT::system_tag;

    __BCinline__ static constexpr int DIMS() { return dimension; };
    __BCinline__ static constexpr int ITERATOR() { return dimension; }

    static_assert(PARENT::ITERATOR() == 0, "RESHAPE IS NOT SUPPORTED ON NON-CONTINUOUS TENSORS");

    scalar_t* array;

    template<class... integers>
    implementation(PARENT parent, BC::array<dimension,int> shape)
    : Shape<dimension>(shape),  array(const_cast<scalar_t*>(parent.memptr())) {}

    __BCinline__ const auto memptr() const { return array; }
    __BCinline__       auto memptr()          { return array; }

    };
};

template<class internal_t, int dimension>
auto make_reshape(internal_t internal, BC::array<dimension,int> shape) {
    return typename Array_Reshape<dimension>::template implementation<internal_t>(internal, shape);
}
}
}

#endif /* TENSOR_RESHAPE_H_ */
