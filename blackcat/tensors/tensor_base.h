/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_TENSOR_BASE_H_
#define BLACKCAT_TENSOR_BASE_H_

#include "tensor_common.h"
#include "expression_templates/tree_evaluator.h"
#include "tensor_iterator_defs.h"
#include "io/print.h"

namespace bc {
namespace tensors {

template<class>
class Tensor_Base;

template<class ExpressionTemplate>
class Expression_Base: public ExpressionTemplate
{

	template<class>
	friend class Tensor_Base;

	using self_type = Expression_Base<ExpressionTemplate>;
	using expression_type = ExpressionTemplate;
	using traits_type = exprs::expression_traits<ExpressionTemplate>;

public:

	static constexpr int tensor_dim = expression_type::tensor_dim;
	static constexpr int tensor_iterator_dim = expression_type::tensor_iterator_dim;

	using value_type  = typename ExpressionTemplate::value_type;
	using system_tag  = typename ExpressionTemplate::system_tag;

	using ExpressionTemplate::ExpressionTemplate;
	using ExpressionTemplate::expression_template;

	//ets are trivially copyable
	Expression_Base(const ExpressionTemplate& et): ExpressionTemplate(et) {}
	Expression_Base(ExpressionTemplate&& et):  ExpressionTemplate(std::move(et)) {}

	#include "tensor_operations.h"

	//Expressions only support element-wise iteration
	//Tensors support column and cwise iteration
	BC_ITERATOR_DEF(cw_, cw_iterator_type, cw_begin, cw_end)
	BC_ITERATOR_DEF(cw_reverse_, cw_reverse_iterator_type, cw_rbegin, cw_rend)
	BC_FORWARD_ITER(cw_, begin, this->expression_template())
	BC_FORWARD_ITER(cw_, end, this->expression_template())

};

template<class ExpressionTemplate>
class Tensor_Base: public Expression_Base<ExpressionTemplate>
{
	template<class>
	friend class Tensor_Base;

	using self_type = Tensor_Base<ExpressionTemplate>;
	using parent_type = Expression_Base<ExpressionTemplate>;
	using expression_type = ExpressionTemplate;
	using traits_type = exprs::expression_traits<ExpressionTemplate>;

public:

	static constexpr int tensor_dim = parent_type::tensor_dim;
	static constexpr int tensor_iterator_dim = parent_type::tensor_iterator_dim;

	using value_type  = typename parent_type::value_type;
	using system_tag  = typename parent_type::system_tag;

	using move_constructible = typename traits_type::is_move_constructible;
	using copy_constructible = typename traits_type::is_move_constructible;
	using move_assignable = typename traits_type::is_move_assignable;
	using copy_assignable = typename traits_type::is_copy_assignable;

	using parent_type::parent_type;
	using parent_type::expression_template;

	Tensor_Base() = default;
	Tensor_Base(const expression_type&  param): parent_type(param) {}
	Tensor_Base(expression_type&& param): parent_type(param) {}

private:

	using tensor_move_type =
			bc::traits::only_if<move_constructible::value, self_type&&>;

	using tensor_copy_type =
			bc::traits::only_if<copy_constructible::value, const self_type&>;
public:

	template<class U>
	Tensor_Base(const Tensor_Base<U>& tensor):
		parent_type(tensor.as_expression_type()) {}

	Tensor_Base(tensor_copy_type tensor):
		parent_type(tensor.as_expression_type()) {}

	Tensor_Base(tensor_move_type tensor):
		parent_type(std::move(tensor.as_expression_type())) {}

	Tensor_Base& operator =(tensor_move_type tensor) noexcept {
		this->as_expression_type() = std::move(tensor.as_expression_type());
		return *this;
	}

	Tensor_Base& operator = (tensor_copy_type param) {
		//From tensor_operations.h"
		assert_valid(param);
		evaluate(this->bi_expr(bc::oper::assign, param));
		return *this;
	}

	Tensor_Base(bc::traits::only_if<tensor_dim==0, value_type> scalar) {
		static_assert(tensor_dim == 0,
				"SCALAR_INITIALIZATION ONLY AVAILABLE TO SCALARS");
		this->fill(scalar);
	}

	BC_FORWARD_ITER(,begin, *this)
	BC_FORWARD_ITER(,end, *this)
	BC_ITERATOR_DEF(,nd_iterator_type, begin, end)
	BC_ITERATOR_DEF(reverse_, nd_reverse_iterator_type, rbegin, rend)

	#include "tensor_utility.h"
	#include "tensor_accessor.h"
	#include "tensor_iteralgos.h"
	#include "tensor_assignment_operations.h"

public:
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

#include "tensor_static_functions.h"

#endif /* TENSOR_BASE_H_ */
