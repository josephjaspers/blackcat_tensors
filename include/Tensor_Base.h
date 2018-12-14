/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef TENSOR_BASE_H_
#define TENSOR_BASE_H_

#include "Tensor_Common.h"

#include "Tensor_Operations.h"
#include "Tensor_Accessor.h"
#include "Tensor_Iterator.h"
#include "Tensor_Algorithm.h"
#include "Tensor_Utility.h"
#include "Tensor_CMath.h"

#include "expression_templates/Array.h"
#include "expression_templates/Array_View.h"
#include "expression_templates/Array_Shared.h"

namespace BC {

template<class internal_t>
class Tensor_Base :
        public internal_t,
        public module::Tensor_Operations<Tensor_Base<internal_t>>,
        public module::Tensor_Algorithm<Tensor_Base<internal_t>>,
        public module::Tensor_Utility<Tensor_Base<internal_t>>,
        public module::Tensor_Accessor<Tensor_Base<internal_t>>,
        public module::Tensor_Iterator<Tensor_Base<internal_t>> {

public:

    using parent        = internal_t;
    using self          = Tensor_Base<internal_t>;
    using operations    = module::Tensor_Operations<Tensor_Base<internal_t>>;
    using utility       = module::Tensor_Utility<Tensor_Base<internal_t>>;
    using accessor      = module::Tensor_Accessor<Tensor_Base<internal_t>>;

    template<class> friend class Tensor_Base;
    using internal_t::internal_t;

    using internal_t::DIMS; //required
    using scalar_t    = typename internal_t::scalar_t;
    using allocator_t = typename internal_t::allocator_t;
    using system_tag  = typename allocator_t::system_tag;

    using operations::operator=;
	using operations::operator+;
	using operations::operator-;
	using operations::operator/;
	using operations::operator*;
	using operations::operator%;
	using operations::operator+=;
	using operations::operator-=;
	using operations::operator/=;
	using operations::operator%=;
	using operations::operator>;
	using operations::operator<;
	using operations::operator>=;
	using operations::operator<=;
	using operations::operator==;

	using accessor::operator[];
    using accessor::operator();

    using move_parameter        = std::conditional_t<et::BC_array_move_constructible<internal_t>(),       self&&, BC::DISABLED<0>>;
    using copy_parameter        = std::conditional_t<et::BC_array_copy_constructible<internal_t>(), const self&,  BC::DISABLED<1>>;
    using move_assign_parameter = std::conditional_t<et::BC_array_move_assignable<internal_t>(),       self&&, BC::DISABLED<0>>;
    using copy_assign_parameter = std::conditional_t<et::BC_array_copy_assignable<internal_t>(), const self&,  BC::DISABLED<1>>;

    Tensor_Base() = default;
    Tensor_Base(const parent&  param) : internal_t(param) {}
    Tensor_Base(
    		parent&& param) : internal_t(param) {}

    template<class U> Tensor_Base(const Tensor_Base<U>&  tensor) : internal_t(tensor.internal()) {}
    template<class U> Tensor_Base(      Tensor_Base<U>&& tensor) : internal_t(tensor.internal()) {}


    Tensor_Base(copy_parameter tensor) {
        this->copy_init(tensor);
    }

    Tensor_Base(move_parameter tensor) {
        this->swap_array(tensor);
        this->swap_shape(tensor);
    }

    Tensor_Base& operator =(move_assign_parameter tensor) {
        this->swap_shape(tensor);
        this->swap_array(tensor);
        return *this;
    }

    Tensor_Base& operator =(copy_assign_parameter tensor) {
         operations::operator=(tensor);
         return *this;
    }

    Tensor_Base(scalar_t scalar) {
        static_assert(DIMS() == 0, "SCALAR_INITIALIZATION ONLY AVAILABLE TO SCALARS");
        this->fill(scalar);
    }

    ~Tensor_Base() {
        this->deallocate();
    }

     const parent& internal() const { return static_cast<const parent&>(*this); }
           parent& internal()       { return static_cast<       parent&>(*this); }

};

}

#endif /* TENSOR_BASE_H_ */
