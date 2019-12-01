/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_TENSOR_BASE_H_
#define BLACKCAT_TENSOR_BASE_H_

#include "Tensor_Common.h"
#include "expression_templates/Tree_Evaluator.h"
#include "io/Print.h"

namespace BC {
namespace tensors {


template<class ExpressionTemplate>
class Tensor_Base: public ExpressionTemplate {

	template<class>
	friend class Tensor_Base;

	using self_type = Tensor_Base<ExpressionTemplate>;
	using expression_type = ExpressionTemplate;
	using traits_type = exprs::expression_traits<ExpressionTemplate>;

public:

	static constexpr int tensor_dimension =
			ExpressionTemplate::tensor_dimension;

	static constexpr int tensor_iterator_dimension =
			ExpressionTemplate::tensor_iterator_dimension;

	using value_type  = typename ExpressionTemplate::value_type;
	using system_tag  = typename ExpressionTemplate::system_tag;

	using move_constructible = typename traits_type::is_move_constructible;
	using copy_constructible = typename traits_type::is_move_constructible;
	using move_assignable = typename traits_type::is_move_assignable;
	using copy_assignable = typename traits_type::is_copy_assignable;

	using ExpressionTemplate::ExpressionTemplate;
	using ExpressionTemplate::internal;

	Tensor_Base() = default;
	Tensor_Base(const expression_type&  param): expression_type(param) {}
	Tensor_Base(expression_type&& param): expression_type(param) {}

private:

	using tensor_move_type =
			BC::traits::only_if<move_constructible::value, self_type&&>;

	using tensor_copy_type =
			BC::traits::only_if<copy_constructible::value, const self_type&>;
public:

	template<class U>
	Tensor_Base(const Tensor_Base<U>& tensor):
		expression_type(tensor.as_expression_type()) {}

	Tensor_Base(tensor_copy_type tensor):
		expression_type(tensor.as_expression_type()) {}

	Tensor_Base(tensor_move_type tensor):
		expression_type(std::move(tensor.as_expression_type())) {}

	Tensor_Base& operator =(BC::traits::only_if<move_assignable::value, self_type&&> tensor) noexcept {
		this->as_expression_type() = std::move(tensor.as_expression_type());
		return *this;
	}

	Tensor_Base(BC::traits::only_if<tensor_dimension==0, value_type> scalar) {
		static_assert(tensor_dimension == 0,
				"SCALAR_INITIALIZATION ONLY AVAILABLE TO SCALARS");
		this->fill(scalar);
	}

#include "Tensor_Utility.h"
#include "Tensor_Accessor.h"
#include "Tensor_IterAlgos.h"
#include "Tensor_Operations.h"

	~Tensor_Base() {
		this->deallocate();
	}

private:

	expression_type& as_expression_type() {
		return static_cast<expression_type&>(*this);
	}
	const expression_type& as_expression_type() const {
		return static_cast<const expression_type&>(*this);
	}
};

}
}

#include "Tensor_Static_Functions.h"

#endif /* TENSOR_BASE_H_ */
