/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_TENSOR_BASE_H_
#define BLACKCAT_TENSOR_BASE_H_

#include "common.h"
#include "expression_templates/tree_evaluator.h"
#include "tensor_iterator_defs.h"
#include "tensor_accessor.h"
#include "expression_base.h"
#include "io/print.h"


namespace bc {
namespace tensors {

template<class ExpressionTemplate>
class Tensor_Base:
	public Expression_Base<ExpressionTemplate>,
	public Tensor_Accessor<ExpressionTemplate>
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

	using Tensor_Accessor<ExpressionTemplate>::operator[];

	Tensor_Base() {};
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
		assert_valid(param);
		evaluate(this->bi_expr(bc::oper::assign, param));
		return *this;
	}

	template<class ValueType, class=std::enable_if_t<std::is_convertible<ValueType, value_type>::value && tensor_dim==0>>
	Tensor_Base(ValueType scalar) {
		this->fill((value_type)scalar);
	}

	BC_FORWARD_ITER(,begin, *this)
	BC_FORWARD_ITER(,end, *this)
	BC_ITERATOR_DEF(,nd_iterator_type, begin, end)
	BC_ITERATOR_DEF(reverse_, nd_reverse_iterator_type, rbegin, rend)

	#include "tensor_utility.h"
	#include "tensor_iteralgos.h"
	#include "tensor_operations.h"

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
