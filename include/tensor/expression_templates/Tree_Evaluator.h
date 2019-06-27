/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PARSE_TREE_COMPLEX_EVALUATOR_H_
#define PARSE_TREE_COMPLEX_EVALUATOR_H_

#include "nd_evaluator/Evaluator.h"
#include "Tree_Evaluator_Optimizer.h"


/*
 *
 * 	This file defines the endpoints for the Tree_Evaluator. The Tree Evaluator
 *  iterates through the tree and attempts use the the left-hand value as a cache (if possible)
 *  and to reduce the need for temporaries when applicable.
 *
 *  After utilizing the left-hand value as cache, the tree is iterated through again, greedy-evaluating
 *  any function calls when possible. A function call is generally any BLAS call.
 *
 *  Example: (assume matrices)
 *  	y += a * b + c * d
 *
 *  	Naively, this expression may generate 2 temporaries, one for each matrix multiplication.
 *  	However, a more efficient way to evaluate this equation would be to make 2 gemm calls,
 *  	gemm(y,a,b) and gemm(y, c, d).
 *
 *  This expression reordering works with more complex calls, such as....
 *  	y = abs(a * b + c * d).
 *
 *  	Here we can apply... (the second gemm call updateing alpha to 1)
 *  	gemm(y,a,b), gemm(y,c,d) followed by evaluating y := abs(y).
 *
 */


namespace BC {
namespace exprs {
namespace detail {

template<class Expression, class SystemTag>
static void nd_evaluate(const Expression expr, BC::Stream<SystemTag> stream) {
	using nd_evaluator  = typename BC::evaluator::Evaluator<SystemTag>;

	BC::meta::constexpr_if<expression_traits<Expression>::is_expr> ([&]() {
		nd_evaluator::template nd_evaluate<Expression::tensor_iterator_dimension>(expr, stream);
	});
}

template<class AssignmentOp, class Left, class Right, class SystemTag>
void greedy_eval(Left left, Right right, BC::Stream<SystemTag> stream) {
	auto temporaries_evaluated_expression = optimizer<Right>::template temporary_injection(right, stream);
	auto assignment_expression = make_bin_expr<AssignmentOp>(left, temporaries_evaluated_expression);
	detail::nd_evaluate(assignment_expression, stream);
	optimizer<decltype(temporaries_evaluated_expression)>::deallocate_temporaries(temporaries_evaluated_expression, stream);
}

}

template<
	class lv,
	class rv,
	class SystemTag,
	class Op>
static
std::enable_if_t<optimizer<Binary_Expression<lv, rv, Op>>::requires_greedy_eval &&
					BC::oper::operation_traits<Op>::is_linear_assignment_operation>
evaluate(Binary_Expression<lv, rv, Op> expression, BC::Stream<SystemTag> stream) {
	static constexpr bool entirely_blas_expression = optimizer<rv>::entirely_blas_expr; // all operations are +/- blas calls
	static constexpr int alpha_mod = BC::oper::operation_traits<Op>::alpha_modifier;
	static constexpr int beta_mod = BC::oper::operation_traits<Op>::beta_modifier;

	auto output = injector<lv, alpha_mod, beta_mod>(expression.left);
	auto right = optimizer<rv>::linear_evaluation(expression.right, output, stream);

	if /*constexpr*/ (!entirely_blas_expression)
		detail::greedy_eval<Op>(expression.left, right, stream);
}

template<
	class lv,
	class rv,
	class SystemTag
	>
static
std::enable_if_t<optimizer<Binary_Expression<lv, rv, oper::Assign>>::requires_greedy_eval>
evaluate(Binary_Expression<lv, rv, oper::Assign> expression, BC::Stream<SystemTag> stream) {
	static constexpr int alpha_mod = BC::oper::operation_traits<oper::Assign>::alpha_modifier; //1
	static constexpr int beta_mod = BC::oper::operation_traits<oper::Assign>::beta_modifier;   //0

	static constexpr bool entirely_blas_expr = optimizer<rv>::entirely_blas_expr;
	static constexpr bool partial_blas_expr = optimizer<rv>::partial_blas_expr && !entirely_blas_expr;

	auto right = optimizer<rv>::injection(expression.right, injector<lv, alpha_mod, beta_mod>(expression.left), stream);

	return BC::meta::constexpr_if<partial_blas_expr>([&]() {
		detail::greedy_eval<oper::Add_Assign>(expression.left, right, stream);
	}, BC::meta::constexpr_else_if<!entirely_blas_expr>([&]() {
		detail::greedy_eval<oper::Assign>(expression.left, right, stream);
	}));
}

template<
	class lv,
	class rv,
	class SystemTag,
	class Op
>
static std::enable_if_t<optimizer<Binary_Expression<lv, rv, Op>>::requires_greedy_eval &&
							!BC::oper::operation_traits<Op>::is_linear_assignment_operation>
evaluate(Binary_Expression<lv, rv, Op> expression, BC::Stream<SystemTag> stream) {
	auto right = optimizer<rv>::temporary_injection(expression.right,  stream);
	detail::greedy_eval<Op>(expression.left, right, stream);
}

template<
	class lv,
	class rv,
	class Op,
	class SystemTag>
static std::enable_if_t<optimizer<Binary_Expression<lv, rv, Op>>::requires_greedy_eval>
evaluate_aliased(Binary_Expression<lv, rv, Op> expression, BC::Stream<SystemTag> stream) {
	detail::greedy_eval<Op>(expression.left, expression.right, stream);
}
//--------------------- lazy only ----------------------- //

//------------------------------------------------Greedy evaluation (BLAS function call detected), skip injection optimization--------------------//
template<class Expression, class Stream>
static std::enable_if_t<!optimizer<Expression>::requires_greedy_eval>
evaluate(Expression expression, Stream stream) {
	detail::nd_evaluate(expression, stream);
}
//------------------------------------------------Purely lazy alias evaluation----------------------------------//
template< class Expression, class Stream>
static std::enable_if_t<!optimizer<Expression>::requires_greedy_eval>
evaluate_aliased(Expression expression, Stream stream_) {
	detail::nd_evaluate(expression, stream_);
}


// ----------------- greedy evaluation --------------------- //

template<class Array, class Expression, class Stream>
auto greedy_evaluate(Array array, Expression expr, Stream stream) {
    static_assert(expression_traits<Array>::is_array, "MAY ONLY EVALUATE TO ARRAYS");
    evaluate(make_bin_expr<oper::Assign>(array, expr), stream);
    return array;
}

//The branch is an array, no evaluation required
template<
	class Expression,
	class Stream,
	class=std::enable_if_t<expression_traits<Expression>::is_array>
>
static auto greedy_evaluate(Expression expression, Stream stream) {
	return expression;
}


template<
	class Expression,
	class Stream,
	class=std::enable_if_t<expression_traits<Expression>::is_expr>,
	int=0
>
static auto greedy_evaluate(Expression expression, Stream stream) {
	/*
	 * Returns a kernel_array containing the tag BC_Temporary,
	 * the caller of the function is responsible for its deallocation.
	 *
	 * Users may query this tag via 'BC::expression_traits<Expression>::is_temporary'
	 */
	using value_type = typename Expression::value_type;
	auto shape = BC::make_shape(expression.inner_shape());
	auto temporary = make_temporary_kernel_array<value_type>(shape, stream);
	return greedy_evaluate(temporary, expression, stream);
}

} //ns exprs
} //ns BC



#endif /* PARSE_TREE_COMPLEX_EVALUATOR_H_ */