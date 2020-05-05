#ifndef BLACKCAT_TENSORS_TENSOR_MASK_PROXY_H_
#define BLACKCAT_TENSORS_TENSOR_MASK_PROXY_H_

#include "expression_base.h"


namespace bc {
namespace tensors {

template<class Parent, class Mask>
class Expression_Mask
{
	template<class>
	friend class Tensor_Base;

	using self_type = Expression_Mask<Parent, Mask>;

public:

	Parent& parent;
	Mask mask;

	Expression_Mask(Parent& parent, Mask mask):
		parent(parent),
		mask(mask) {}

	static constexpr int tensor_dim = Parent::tensor_dim;
	static constexpr int tensor_iterator_dim = Parent::tensor_iterator_dim;

	using value_type  = typename Parent::value_type;
	using system_tag  = typename Parent::system_tag;


	template<class ExpressionTemplate>
	Parent& operator = (const Expression_Base<ExpressionTemplate>& other)
	{
		return evaluate_masked_expression(bc::oper::assign, other);
	}

	template<class ExpressionTemplate>
	Parent& operator += (const Expression_Base<ExpressionTemplate>& other)
	{
		return evaluate_masked_expression(bc::oper::add_assign, other);
	}

	template<class ExpressionTemplate>
	Parent& operator -= (const Expression_Base<ExpressionTemplate>& other)
	{
		return evaluate_masked_expression(bc::oper::sub_assign, other);
	}

	template<class ExpressionTemplate>
	Parent& operator /= (const Expression_Base<ExpressionTemplate>& other)
	{
		return evaluate_masked_expression(bc::oper::div_assign, other);
	}

	template<class ExpressionTemplate>
	Parent& operator %= (const Expression_Base<ExpressionTemplate>& other)
	{
		return evaluate_masked_expression(bc::oper::mul_assign, other);
	}

private:

	template<class AssignmentFunc, class ExpressionTemplate>
	Parent& evaluate_masked_expression(
		AssignmentFunc func,
		const Expression_Base<ExpressionTemplate>& other)
	{
		return evaluate(make_masked_expression(func, other));
	}

	template<class AssignmentFunc, class ExpressionTemplate>
	auto make_masked_expression(
		AssignmentFunc func,
		const Expression_Base<ExpressionTemplate>& other)
	{
		auto parent_et = parent.expression_template();
		auto assign_et = other.expression_template();
		auto mask_et   = mask.expression_template();

		auto assign_expression = bc::tensors::exprs::make_bin_expr(
			parent_et,
			assign_et,
			func);

		auto masked_expression = bc::tensors::exprs::make_bin_expr(
			assign_expression,
			mask_et,
			bc::tensors::exprs::Mask()
		);

		return masked_expression;
	}

	template<class Xpr>
	Parent& evaluate(const Xpr& expression) {
		auto& allocator = parent.get_stream().get_allocator();

		BC_ASSERT(allocator.allocated_bytes() == 0,
			"Evaluation expects streams allocate_bytes to be 0 pre-evaluation");

		exprs::evaluate(expression, parent.get_stream());

		BC_ASSERT(allocator.allocated_bytes() == 0,
			"Evaluation expects streams allocate_bytes to be 0 post-evaluation");
		return parent;
	}
};

template<class Parent, class Mask>
auto make_expression_mask(Parent& parent, Mask mask)
{
	return Expression_Mask<Parent, Mask> { parent, mask };
}

}
}

#endif
