/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_BLACKCAT_COMMON_H_
#define BLACKCAT_BLACKCAT_COMMON_H_

#include <type_traits>

#include "expression_templates/expression_templates.h"
#include "iterators/iterators.h"

namespace bc {
namespace tensors {

template<int X>
struct Tensor_Dim : bc::traits::Integer<X> {
	static constexpr int tensor_dim = X;
};

template<class ExpressionTemplate>
class Expression_Base;

template<class ExpressionTemplate>
class Tensor_Base;

template<class ExpressionTemplate>
auto make_tensor(ExpressionTemplate expression) 
{
	static_assert(
		exprs::expression_traits<ExpressionTemplate>
				::is_expression_template::value,
		"Make Tensor can only be used with Expression_Template");
	return Tensor_Base<ExpressionTemplate>(expression);
}

template<class ExpressionTemplate>
class Expression_Base;

template<class ExpressionTemplate>
auto make_expression(ExpressionTemplate expression) {
	static_assert(
		exprs::expression_traits<ExpressionTemplate>
				::is_expression_template::value,
		"Make Tensor can only be used with Expression_Template");
	return Expression_Base<ExpressionTemplate>(expression);
}


template<
	class ExpressionTemplate,
	class Allocator = bc::Allocator<
		typename exprs::expression_traits<ExpressionTemplate>::value_type,
		typename exprs::expression_traits<ExpressionTemplate>::system_tag>>
auto evaluate(ExpressionTemplate expression, Allocator allocator = Allocator())
{
	constexpr
	int tensor_dim   = ExpressionTemplate::tensor_dim;

	using traits     = exprs::expression_traits<ExpressionTemplate>;
	using value_type = typename traits::value_type;
	using system_tag = typename traits::system_tag;

	using Tensor = Tensor_Base<exprs::Array<
			bc::Shape<tensor_dim>,
			value_type,
			Allocator>>;
}

}
}

#endif /* BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_ */
