/*
 * expression_base.h
 *
 *  Created on: Apr 14, 2020
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_EXPRESSION_BASE_H_
#define BLACKCAT_TENSORS_EXPRESSION_BASE_H_

#include "common.h"
#include "expression_templates/array.h"
#include "expression_templates/tree_evaluator.h"
#include "tensor_iterator_defs.h"

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
	Expression_Base() {}
	Expression_Base(const ExpressionTemplate& et): ExpressionTemplate(et) {}
	Expression_Base(ExpressionTemplate&& et):  ExpressionTemplate(std::move(et)) {}

	//Expressions only support element-wise iteration
	//Tensors support column and cwise iteration
	BC_ITERATOR_DEF(cw_, cw_iterator_type, cw_begin, cw_end)
	BC_ITERATOR_DEF(cw_reverse_, cw_reverse_iterator_type, cw_rbegin, cw_rend)
	BC_FORWARD_ITER(cw_, begin, this->expression_template())
	BC_FORWARD_ITER(cw_, end, this->expression_template())

	template<int ADL=0>
	std::string to_string() const
	{
		using tensor_type = Tensor_Base<exprs::Array<
			Shape<tensor_dim>,
			value_type,
			bc::Allocator<value_type, system_tag>>>;
		return tensor_type(*this).to_string();
	}

	#include "expression_operations.inl"
	#include "expression_utility.inl"

	friend std::ostream& operator << (std::ostream& os, const Expression_Base<ExpressionTemplate>& self)
	{
		return os << self.to_string();
	}
};

}
}


#endif
