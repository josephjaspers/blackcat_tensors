/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_RAND_H_
#define BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_RAND_H_

#include "Array_Base.h"


namespace BC {
namespace et     {
//identical to Array_Scalar, though the scalar is allocated on the stack opposed to heap
template<class scalar_t_, class allocator_t_>
struct Rand_Constant : Shape<0>, Array_Base<Rand_Constant<scalar_t_, allocator_t_>, 0>{

    using scalar_t = scalar_t_;
    using allocator_t = allocator_t_;
    using mathlib_t = typename allocator_t::mathlib_t;

    __BCinline__ static constexpr int ITERATOR() { return 0; }
    __BCinline__ static constexpr int DIMS()      { return 0; }

    operator scalar_t () { return value(); }

    scalar_t lower_bound, upper_bound;

    scalar_t value() const {
        return allocator_t::rand(lower_bound, upper_bound);
    }



    Rand_Constant(scalar_t lower_bound_, scalar_t upper_bound_)
    : lower_bound(lower_bound_), upper_bound(upper_bound_) {}


    template<class... integers>
    auto operator()  (const integers&...) const{ return value(); }
    auto operator [] (int i ) { return value(); }

    const scalar_t* memptr() const { return nullptr; }
};
}
}

#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_RAND_H_ */
