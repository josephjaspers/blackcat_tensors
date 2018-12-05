/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_
#define BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_

#include "Array_Base.h"

namespace BC {
namespace et {

//identical to Array_Scalar, though the scalar is allocated on the stack opposed to heap
template<class scalar_t_, class allocator_t_>
struct Scalar_Constant : Shape<0>, Array_Base<Scalar_Constant<scalar_t_, allocator_t_>, 0>{

    using scalar_t = scalar_t_;
    using allocator_t = allocator_t_;
    using system_tag = typename allocator_t_::system_tag;

    __BCinline__ static constexpr int ITERATOR() { return 0; }
    __BCinline__ static constexpr int DIMS()      { return 0; }

    scalar_t scalar;

    __BCinline__ operator scalar_t () const {
        return scalar;
    }

    __BCinline__ Scalar_Constant(scalar_t scalar_) : scalar(scalar_) {}


    template<class... integers> __BCinline__ auto operator()  (const integers&...) const{ return scalar; }
    template<class... integers> __BCinline__ auto operator()  (const integers&...) { return scalar; }

    __BCinline__ auto operator [] (int i ) const { return scalar; }
    __BCinline__ auto operator [] (int i )  { return scalar; }

    __BCinline__ const scalar_t* memptr() const { return &scalar; }

    void swap_array(Scalar_Constant&) {}
};

template<class allocator_t, class scalar_t>
auto scalar_constant(scalar_t scalar) {
    return Scalar_Constant<scalar_t, allocator_t>(scalar);
}
}
}




#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_ */
