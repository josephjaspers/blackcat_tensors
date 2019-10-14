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


/** @File
 *
 * 	This file defines the end-points for the Tree_Evaluator. The Tree Evaluator
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
namespace tensors {
namespace exprs { 
namespace detail {

template<class Expression, class Stream>
static void nd_evaluate(const Expression expression, Stream stream) {
	using system_tag	= typename Stream::system_tag;
	using nd_evaluator  = typename BC::tensors::exprs::evaluator::Evaluator<system_tag>;

	static_assert(Expression::tensor_iterator_dimension >= Expression::tensor_dimension
			|| Expression::tensor_iterator_dimension <= 1,
			"Iterator Dimension must be greater than or equal to the tensor_dimension");

	BC::traits::constexpr_if<expression_traits<Expression>::is_expr::value> ([&]() {
		nd_evaluator::template nd_evaluate<Expression::tensor_iterator_dimension>(expression, stream);
	});
}

template<class AssignmentOp, class Left, class Right, class Stream, class TruthType>
void greedy_optimization(Left left, Right right, Stream stream, TruthType is_subexpression) {
	auto temporaries_evaluated_expression = optimizer<Right>::template temporary_injection(right, stream);
	auto assignment_expression = make_bin_expr<AssignmentOp>(left, temporaries_evaluated_expression);
	detail::nd_evaluate(assignment_expression, stream);

	if (!TruthType::value)
	optimizer<decltype(temporaries_evaluated_expression)>::deallocate_temporaries(temporaries_evaluated_expression, stream);
}

template<
	class lv,
	class rv,
	class Stream,
	class Op,
	class TruthType=std::false_type>
static
std::enable_if_t<optimizer<Binary_Expression<Op, lv, rv>>::requires_greedy_eval &&
					BC::oper::operation_traits<Op>::is_linear_assignment_operation>
evaluate(Binary_Expression<Op, lv, rv> expression, Stream stream, TruthType is_subexpression=TruthType()) {
	static constexpr bool entirely_blas_expression = optimizer<rv>::entirely_blas_expr; // all operations are +/- blas calls
	static constexpr int alpha_mod = BC::oper::operation_traits<Op>::alpha_modifier;
	static constexpr int beta_mod = BC::oper::operation_traits<Op>::beta_modifier;

	auto output = make_output_data<alpha_mod, beta_mod>(expression.left);
	auto right = optimizer<rv>::linear_evaluation(expression.right, output, stream);

	if /*constexpr*/ (!entirely_blas_expression)
		detail::greedy_optimization<Op>(expression.left, right, stream, is_subexpression);
	else if (!TruthType::value) {
		optimizer<decltype(right)>::deallocate_temporaries(right, stream);
	}
}

template<
	class lv,
	class rv,
	class Stream,
	class TruthType=std::false_type
	>
static
std::enable_if_t<optimizer<Binary_Expression<oper::Assign, lv, rv>>::requires_greedy_eval>
evaluate(Binary_Expression<oper::Assign, lv, rv> expression, Stream stream, TruthType is_subexpression=TruthType()) {
	static constexpr int alpha = BC::oper::operation_traits<oper::Assign>::alpha_modifier; //1
	static constexpr int beta = BC::oper::operation_traits<oper::Assign>::beta_modifier;   //0
	static constexpr bool entirely_blas_expr = optimizer<rv>::entirely_blas_expr;

	auto output = make_output_data<alpha, beta>(expression.left);
	auto right = optimizer<rv>::injection(expression.right, output, stream);

	BC::traits::constexpr_if<!entirely_blas_expr>([&]() {
		detail::greedy_optimization<oper::Assign>(expression.left, right, stream, is_subexpression);
	});
}

template<
	class lv,
	class rv,
	class Stream,
	class Op,
	class TruthType
>
static std::enable_if_t<optimizer<Binary_Expression<Op, lv, rv>>::requires_greedy_eval &&
							!BC::oper::operation_traits<Op>::is_linear_assignment_operation>
evaluate(Binary_Expression<Op, lv, rv> expression, Stream stream,  TruthType is_subexpression=TruthType()) {
	auto right = optimizer<rv>::temporary_injection(expression.right, stream);
	detail::greedy_optimization<Op>(expression.left, right, stream, is_subexpression);
}

template<
	class lv,
	class rv,
	class Op,
	class Stream>
static std::enable_if_t<optimizer<Binary_Expression<Op, lv, rv>>::requires_greedy_eval>
evaluate_aliased(Binary_Expression<Op, lv, rv> expression, Stream stream) {
	detail::greedy_optimization<Op>(expression.left, expression.right, stream, std::false_type());
}
//--------------------- lazy only ----------------------- //

//------------------------------------------------Greedy evaluation (BLAS function call detected), skip injection optimization--------------------//
template<class Expression, class Stream, class TruthType=std::false_type>
static std::enable_if_t<!optimizer<Expression>::requires_greedy_eval>
evaluate(Expression expression, Stream stream, TruthType=TruthType()) {
	detail::nd_evaluate(expression, stream);
}
//------------------------------------------------Purely lazy alias evaluation----------------------------------//
template< class Expression, class Stream>
static std::enable_if_t<!optimizer<Expression>::requires_greedy_eval>
evaluate_aliased(Expression expression, Stream stream) {
	detail::nd_evaluate(expression, stream);
}


// ----------------- greedy evaluation --------------------- //


//The branch is an array, no evaluation required
template<
	class Expression,
	class Stream,
	class=std::enable_if_t<expression_traits<Expression>::is_array::value>
>
static auto greedy_evaluate(Expression expression, Stream stream) {
	return expression;
}


template<
	class Expression,
	class Stream,
	class=std::enable_if_t<expression_traits<Expression>::is_expr::value>,
	int=0
>
static auto greedy_evaluate(Expression expression, Stream stream) {
	/*
	 * Returns a kernel_array containing the tag temporary_tag,
	 * the caller of the function is responsible for its deallocation.
	 *
	 * Users may query this tag via 'BC::expression_traits<Expression>::is_temporary'
	 */
	using value_type = typename Expression::value_type;
	auto temporary = make_kernel_array(
			expression.get_shape(),
			stream.template get_allocator_rebound<value_type>(),
			temporary_tag());

	detail::evaluate(make_bin_expr<oper::Assign>(temporary, expression), stream, std::true_type());
	return temporary;
}


} //ns detail


//----------------------------------- endpoints ---------------------------------------//
template<class Expression, class Stream>
static auto greedy_evaluate(Expression expression, Stream stream) {
	if (optimizer<Expression>::requires_greedy_eval) {
		//Initialize a logging_stream (does not call any jobs-enqueued or allocate memory, simply logs memory requirements)
		BC::streams::Logging_Stream<typename Stream::system_tag> logging_stream;
		detail::greedy_evaluate(expression, logging_stream);	//record allocations/deallocations
		stream.get_allocator().reserve(logging_stream.get_max_allocated());	//Reserve the maximum amount of memory
	}
	return detail::greedy_evaluate(expression, stream);	//Do the actual calculation
}


template<class Expression, class Stream>
static auto evaluate(Expression expression, Stream stream) {
	if (optimizer<Expression>::requires_greedy_eval) {
		BC::streams::Logging_Stream<typename Stream::system_tag> logging_stream;
		detail::evaluate(expression, logging_stream);
		stream.get_allocator().reserve(logging_stream.get_max_allocated());
	}
	return detail::evaluate(expression, stream);
}

template<class Expression, class Stream>
static auto evaluate_aliased(Expression expression, Stream stream) {
	if (optimizer<Expression>::requires_greedy_eval) {
		BC::streams::Logging_Stream<typename Stream::system_tag> logging_stream;
		detail::evaluate_aliased(expression, logging_stream);
		stream.get_allocator().reserve(logging_stream.get_max_allocated());
	}
	return detail::evaluate_aliased(expression, stream);
}

template<class Expression, class SystemTag>
static auto greedy_evaluate(Expression expression, BC::streams::Logging_Stream<SystemTag> logging_stream) {
	return detail::greedy_evaluate(expression, logging_stream);	//record allocations/deallocations
}

template<class Expression, class SystemTag>
static auto evaluate(Expression expression, BC::streams::Logging_Stream<SystemTag> logging_stream) {
	return detail::evaluate(expression, logging_stream);
}

template<class Expression, class SystemTag>
static auto evaluate_aliased(Expression expression, BC::streams::Logging_Stream<SystemTag> logging_stream) {
	return detail::evaluate_aliased(expression, logging_stream);
}




} //ns exprs
} //ns tensors
} //ns BC


#endif /* PARSE_TREE_COMPLEX_EVALUATOR_H_ */
