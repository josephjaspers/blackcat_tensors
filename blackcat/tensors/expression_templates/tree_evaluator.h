/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PARSE_TREE_COMPLEX_EVALUATOR_H_
#define PARSE_TREE_COMPLEX_EVALUATOR_H_

#include "nd_evaluator/evaluator.h"
#include "tree_evaluator_optimizer.h"


/** @File
 *  The Evaluator determines if an expression needs to be greedily optimized.
 *  If it attempts to use the left-hand variable (the output) as cache to
 *  reduce temporaries as possible. Then, if functions still requiring
 *  evaluating will use temporaries to replace the functions and finally
 *  complete the evaluation of the expresssion with an elementwise (nd_evaluator).
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
namespace bc {
namespace tensors {
namespace exprs { 

template<class Is_SubXpr=std::false_type>
class Evaluator
{
	template<class Xpr, class Stream>
	static void nd_evaluate(const Xpr expression, Stream stream)
	{
		using system_tag	= typename Stream::system_tag;
		using nd_evaluator  = exprs::evaluator::Evaluator<system_tag>;

		bc::traits::constexpr_if<expression_traits<Xpr>::is_expr::value>([&]() {
			nd_evaluator::template nd_evaluate<Xpr::tensor_iterator_dim>(expression, stream);
		});
	}

	template<class AssignmentOp, class Left, class Right, class Stream>
	static void greedy_optimization(Left left, Right right, Stream stream)
	{
		auto right_xpr = optimizer<Right>::temporary_injection(right, stream);
		auto assign_xpr = make_bin_expr<AssignmentOp>(left, right_xpr);
		nd_evaluate(assign_xpr, stream);

		if (!Is_SubXpr::value) {
			using right_xpr_t = std::decay_t<decltype(right_xpr)>;
			optimizer<right_xpr_t>::deallocate_temporaries(right_xpr, stream);
		}
	}

public:
	// y += blas_op(x1) +/- blas_op(x2) OR y -= blas_op(x1) +/- blas_op(x2)
	template<class lv, class rv, class Stream, class Op>
	static std::enable_if_t<
			optimizer<Bin_Op<Op, lv, rv>>::requires_greedy_eval &&
			bc::oper::operation_traits<Op>::is_linear_assignment_operation>
	evaluate(Bin_Op<Op, lv, rv> expression, Stream stream)
	{
		static constexpr bool entirely_blas_expression = optimizer<rv>::entirely_blas_expr; // all operations are +/- blas calls
		static constexpr int alpha_mod = bc::oper::operation_traits<Op>::alpha_modifier;
		static constexpr int beta_mod = bc::oper::operation_traits<Op>::beta_modifier;

		auto output = make_output_data<alpha_mod, beta_mod>(expression.left);
		auto right = optimizer<rv>::linear_eval(expression.right, output, stream);

		if /*constexpr*/ (!entirely_blas_expression)
			greedy_optimization<Op>(expression.left, right, stream);
		else if (!Is_SubXpr::value) {
			optimizer<decltype(right)>::deallocate_temporaries(right, stream);
		}
	}

	// y = <expression with at least 1 blas_op>
	template<class lv, class rv, class Stream>
	static std::enable_if_t<
			optimizer<Bin_Op<bc::oper::Assign, lv, rv>>::requires_greedy_eval>
	evaluate(Bin_Op<oper::Assign, lv, rv> expression, Stream stream)
	{
		constexpr int alpha = bc::oper::operation_traits<oper::Assign>::alpha_modifier; //1
		constexpr int beta = bc::oper::operation_traits<oper::Assign>::beta_modifier;   //0
		constexpr bool entirely_blas_expr = optimizer<rv>::entirely_blas_expr;
		constexpr bool partial_blas_expr = optimizer<rv>::partial_blas_expr;

		auto output = make_output_data<alpha, beta>(expression.left);
		using expr_rv_t = std::decay_t<decltype(expression.right)>;

		auto right = bc::traits::constexpr_ternary<partial_blas_expr>(
			[&]() {
				return optimizer<expr_rv_t>::linear_eval(
						expression.right, output, stream);
			},
			[&]() {
				return optimizer<expr_rv_t>::injection(
						expression.right, output, stream);
			});

		using assignment_oper = std::conditional_t<
				partial_blas_expr, oper::Add_Assign, oper::Assign>;

		bc::traits::constexpr_if<!entirely_blas_expr>([&]() {
			greedy_optimization<assignment_oper>(expression.left, right, stream);
		});
	}

	// y %= <expression> OR y /= <expression>
	template<class lv, class rv, class Stream, class Op>
	static std::enable_if_t<
			optimizer<Bin_Op<Op, lv, rv>>::requires_greedy_eval &&
			!bc::oper::operation_traits<Op>::is_linear_assignment_operation>
	evaluate(Bin_Op<Op, lv, rv> expression, Stream stream) {
		greedy_optimization<Op>(expression.left, expression.right, stream);
	}

	// y <any assignment op> <elementwise-only expression>
	template<class Xpr, class Stream>
	static std::enable_if_t<!optimizer<Xpr>::requires_greedy_eval>
	evaluate(Xpr expression, Stream stream) {
		nd_evaluate(expression, stream);
	}
};

struct GreedyEvaluator {

	template<
		class Xpr,
		class Stream,
		class=std::enable_if_t<expression_traits<Xpr>::is_array::value>>
	static auto evaluate(Xpr expression, Stream stream) {
		return expression;
	}

	/**
	 * Returns a kernel_array containing the tag temporary_tag,
	 * the caller of the function is responsible for its deallocation.
	 * query this tag via 'exprs::expression_traits<Xpr>::is_temporary'
	 */
	template<
		class Xpr,
		class Stream,
		class=std::enable_if_t<expression_traits<Xpr>::is_expr::value>,
		int=0>
	static auto evaluate(Xpr expression, Stream stream)
	{
		using value_type = typename Xpr::value_type;
		auto allocator = stream.template get_allocator_rebound<value_type>();
		auto shape = expression.get_shape();
		auto temporary = make_kernel_array(shape, allocator, temporary_tag());

		Evaluator<std::true_type>::evaluate(
				make_bin_expr<oper::Assign>(temporary, expression), stream);
		return temporary;
	}
};

// ------------------------------ endpoints ------------------------------//
template<class Xpr, class Stream>
static auto greedy_evaluate(Xpr expression, Stream stream) {
	if (optimizer<Xpr>::requires_greedy_eval) {
		//Initialize a logging_stream (does not call any jobs-enqueued or allocate memory, simply logs memory requirements)
		bc::streams::Logging_Stream<typename Stream::system_tag> logging_stream;
		GreedyEvaluator::evaluate(expression, logging_stream);	//record allocations/deallocations
		stream.get_allocator().reserve(logging_stream.get_max_allocated());	//Reserve the maximum amount of memory
	}
	return GreedyEvaluator::evaluate(expression, stream);	//Do the actual calculation
}

template<class Xpr, class Stream>
static auto evaluate(Xpr expression, Stream stream) {
	if (optimizer<Xpr>::requires_greedy_eval) {
		bc::streams::Logging_Stream<typename Stream::system_tag> logging_stream;
		Evaluator<>::evaluate(expression, logging_stream);
		stream.get_allocator().reserve(logging_stream.get_max_allocated());
	}

	return Evaluator<>::evaluate(expression, stream);
}

template<class Xpr, class SystemTag>
static auto greedy_evaluate(
		Xpr expression, bc::streams::Logging_Stream<SystemTag> logging_stream) {
	return GreedyEvaluator::evaluate(expression, logging_stream);
}

template<class Xpr, class SystemTag>
static auto evaluate(
		Xpr expression, bc::streams::Logging_Stream<SystemTag> logging_stream) {
	return Evaluator<>::evaluate(expression, logging_stream);
}

} //ns exprs
} //ns tensors
} //ns BC

#endif /* PARSE_TREE_COMPLEX_EVALUATOR_H_ */
