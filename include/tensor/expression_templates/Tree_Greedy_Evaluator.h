/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PARSE_TREE_COMPLEX_EVALUATOR_H_
#define PARSE_TREE_COMPLEX_EVALUATOR_H_

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
namespace tree {
namespace evaluator_paths {
/*
 * These overloads determine the initial alpha and beta modifiers.
 * (The integer template values of the 'injector' class)
 */

template<class expression_t, class SystemTag>
static auto evaluate_temporaries(expression_t expression, BC::Stream<SystemTag>& stream) {
	return BC::meta::constexpr_if<optimizer<expression_t>::requires_greedy_eval>([&]() {
		return optimizer<expression_t>::template temporary_injection(expression, stream);
	}, [&]() {
		return expression;
	});
}

template<
	class lv,
	class rv,
	class SystemTag,
	class Op,
	class=std::enable_if_t<BC::oper::operation_traits<Op>::is_linear_assignment_operation>> // += or -=
static  auto evaluate(Binary_Expression<lv, rv, Op> expression, BC::Stream<SystemTag> stream) {
	static constexpr bool entirely_blas_expression = optimizer<rv>::entirely_blas_expr; // all operations are +/- blas calls
	static constexpr int alpha_mod = BC::oper::operation_traits<Op>::alpha_modifier;
	static constexpr int beta_mod = BC::oper::operation_traits<Op>::beta_modifier;

	return BC::meta::constexpr_ternary<entirely_blas_expression>([&]() {
		optimizer<rv>::linear_evaluation(expression.right, injector<lv, alpha_mod, beta_mod>(expression.left), stream);
		return expression.left;
	}, [&]() {
		auto right = optimizer<rv>::linear_evaluation(expression.right, injector<lv, alpha_mod, beta_mod>(expression.left), stream);
		return make_bin_expr<Op>(expression.left, evaluate_temporaries(right, stream));
	});
}

template<
	class lv,
	class rv,
	class SystemTag>
static  auto evaluate(Binary_Expression<lv, rv, oper::Assign> expression, BC::Stream<SystemTag> stream) {
	static constexpr int alpha_mod = BC::oper::operation_traits<oper::Assign>::alpha_modifier; //1
	static constexpr int beta_mod = BC::oper::operation_traits<oper::Assign>::beta_modifier;   //0
	auto right = optimizer<rv>::injection(expression.right, injector<lv, alpha_mod, beta_mod>(expression.left), stream);

	return BC::meta::constexpr_if<optimizer<rv>::partial_blas_expr && !optimizer<rv>::entirely_blas_expr>([&]() {
		return make_bin_expr<oper::Add_Assign>(expression.left, evaluate_temporaries(right, stream));

	}, BC::meta::constexpr_else_if<optimizer<rv>::entirely_blas_expr>([&]() {
		return expression.left;

	}, [&]() { //else !optimizer<rv>::partial_blas_expr && !optimizer<rv>::entirely_blas_expr
		return make_bin_expr<oper::Assign>(expression.left, evaluate_temporaries(right, stream));
	}));
}

template<
	class lv,
	class rv,
	class SystemTag,
	class assignment_oper,
	class=std::enable_if_t<!BC::oper::operation_traits<assignment_oper>::is_linear_assignment_operation>,
	int ADL=0
>
static auto evaluate(Binary_Expression<lv, rv, assignment_oper> expression, BC::Stream<SystemTag> stream) {
	auto right_eval =  optimizer<rv>::temporary_injection(expression.right,  stream);
	return make_bin_expr<assignment_oper>(expression.left, right_eval);
}

template<class lv, class rv, class op, class SystemTag>
static auto evaluate_aliased(Binary_Expression<lv, rv, op> expression, BC::Stream<SystemTag> stream) {
	auto right = evaluate_temporaries(expression.right, stream);
	return make_bin_expr<op>(expression.left, right);
}

template<class expression, class SystemTag>
static void deallocate_temporaries(expression expr, BC::Stream<SystemTag>& stream) {
	optimizer<expression>::deallocate_temporaries(expr, stream);
}

} //ns evaluator_paths
} //ns tree
} //ns exprs
} //ns BC



#endif /* PARSE_TREE_COMPLEX_EVALUATOR_H_ */
