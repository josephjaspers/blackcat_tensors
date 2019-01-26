/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_TENSOR_BASE_H_
#define BLACKCAT_TENSOR_BASE_H_


#include "Tensor_Operations.h"
#include "Tensor_Accessor.h"
#include "Tensor_Iterator.h"
#include "Tensor_Algorithm.h"
#include "Tensor_Utility.h"
#include "Tensor_CMath.h"

#include "expression_templates/Array.h"
#include "expression_templates/Array_View.h"

namespace BC {

template<class ExpressionTemplate>
class Tensor_Base :
        public ExpressionTemplate,
        public module::Tensor_Operations<Tensor_Base<ExpressionTemplate>>,
        public module::Tensor_Algorithm<Tensor_Base<ExpressionTemplate>>,
        public module::Tensor_Utility<Tensor_Base<ExpressionTemplate>>,
        public module::Tensor_Accessor<Tensor_Base<ExpressionTemplate>>,
        public module::Tensor_Iterator<Tensor_Base<ExpressionTemplate>> {

public:

    using parent        = ExpressionTemplate;
    using self          = Tensor_Base<ExpressionTemplate>;
    using operations    = module::Tensor_Operations<Tensor_Base<ExpressionTemplate>>;
    using utility       = module::Tensor_Utility<Tensor_Base<ExpressionTemplate>>;
    using accessor      = module::Tensor_Accessor<Tensor_Base<ExpressionTemplate>>;

    friend class module::Tensor_Operations<Tensor_Base<ExpressionTemplate>>;
    friend class module::Tensor_Utility<Tensor_Base<ExpressionTemplate>>;
    friend class module::Tensor_Accessor<Tensor_Base<ExpressionTemplate>>;

    template<class> friend class Tensor_Base;
    using ExpressionTemplate::ExpressionTemplate;
    using ExpressionTemplate::internal;

    using ExpressionTemplate::DIMS; //required
    using value_type  = typename ExpressionTemplate::value_type;
    using allocator_t = typename ExpressionTemplate::allocator_t;
	using system_tag  = typename BC::allocator_traits<allocator_t>::system_tag;

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

    using move_parameter        = BC::meta::only_if<BC::is_move_constructible_v<ExpressionTemplate>, self&&>;
    using copy_parameter        = BC::meta::only_if<BC::is_copy_constructible_v<ExpressionTemplate>, const self&>;
    using move_assign_parameter = BC::meta::only_if<BC::is_move_assignable_v<ExpressionTemplate>,       self&&>;
    using copy_assign_parameter = BC::meta::only_if<BC::is_copy_assignable_v<ExpressionTemplate>, const self&>;

    Tensor_Base() = default;
    Tensor_Base(const parent&  param) : parent(param) {}
    Tensor_Base(parent&& param) : parent(param) {}

    template<class U> Tensor_Base(const Tensor_Base<U>&  tensor) : parent(tensor.as_parent()) {}


    Tensor_Base(copy_parameter tensor)
    : parent(tensor.as_parent()) {}

    Tensor_Base(move_parameter tensor)
    : parent(std::move(tensor.as_parent())) {}

    Tensor_Base& operator =(move_assign_parameter tensor) {
        this->internal_swap(tensor.as_parent());
        return *this;
    }

    Tensor_Base& operator =(copy_assign_parameter tensor) {
         operations::operator=(tensor);
         return *this;
    }

    Tensor_Base(value_type scalar) {
        static_assert(DIMS == 0, "SCALAR_INITIALIZATION ONLY AVAILABLE TO SCALARS");
        this->fill(scalar);
    }

    ~Tensor_Base() {
        this->deallocate();
    }


private:

    parent& as_parent() {
    	return static_cast<parent&>(*this);
    }
    const parent& as_parent() const {
    	return static_cast<const parent&>(*this);
    }
};

}

#endif /* TENSOR_BASE_H_ */
