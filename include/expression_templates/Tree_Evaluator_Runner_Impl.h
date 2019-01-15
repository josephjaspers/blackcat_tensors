/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PARSE_TREE_COMPLEX_EVALUATOR_H_
#define PARSE_TREE_COMPLEX_EVALUATOR_H_

#include "Tree_Evaluator.h"


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
 *  	Naively, this expression may generate 2 temporaries, one for each matrix multplication.
 *  	However, a more efficient way to evaluate this equation would be to make 2 gemm calls,
 *  	gemm(y,a,b) and gemm(y, c, d).
 *
 *  This expression reordering works with more complex calls, such as....
 *  	y = abs(a * b + c * d).
 *
 *  	Here we can apply... (the second gemm call updateing alpha to 1)
 *  	gemm(y,a,b), gemm(y,c,d) followed by evaluationg y := abs(y).
 *
 */


namespace BC {
namespace et {
namespace tree {


struct Greedy_Evaluator {

	template<class lv, class rv, class Allocator>
	static  auto evaluate(Binary_Expression<lv, rv, oper::add_assign> expression, Allocator& alloc) {
		using allocator_t = typename lv::allocator_t;

		auto right = evaluator<rv>::linear_evaluation(expression.right, injector<lv, 1, 1>(expression.left), alloc);
		return make_bin_expr<oper::add_assign>(expression.left, evaluate_temporaries<allocator_t>(right, alloc));
	}

	template<class lv, class rv, class Allocator>
	static auto evaluate(Binary_Expression<lv, rv, oper::sub_assign> expression, Allocator& alloc) {
		using allocator_t = typename lv::allocator_t;

		auto right = evaluator<rv>::linear_evaluation(expression.right, injector<lv, -1, 1>(expression.left), alloc);
		return make_bin_expr<oper::sub_assign>(expression.left, evaluate_temporaries<allocator_t>(right, alloc));
	}

	template<class lv, class rv, class Allocator, class=std::enable_if_t<!evaluator<rv>::partial_blas_expr>>
	static auto evaluate(Binary_Expression<lv, rv, oper::assign> expression, Allocator& alloc) {
		using allocator_t = typename lv::allocator_t;

		auto right = evaluator<rv>::injection(expression.right, injector<lv, 1, 0>(expression.left), alloc);
		return make_bin_expr<oper::assign>(expression.left, evaluate_temporaries<allocator_t>(right, alloc));
	}

	template<class lv, class rv, class Allocator, class=std::enable_if_t<evaluator<rv>::partial_blas_expr>, int=0>
	static auto evaluate(Binary_Expression<lv, rv, oper::assign> expression, Allocator& alloc) {
		using allocator_t = typename lv::allocator_t;

		auto right = evaluator<rv>::linear_evaluation(expression.right, injector<lv, 1, 0>(expression.left), alloc);
		return make_bin_expr<oper::add_assign>(expression.left, evaluate_temporaries<allocator_t>(right, alloc));
	}


	template<class lv, class rv, class Allocator>
	static auto evaluate(Binary_Expression<lv, rv, oper::mul_assign> expression, Allocator& alloc) {
		auto right_eval =  evaluator<rv>::temporary_injection(expression.right,  alloc);
		return make_bin_expr<oper::mul_assign>(expression.left, right_eval);

	}

	template<class lv, class rv, class Allocator>
	static auto evaluate(Binary_Expression<lv, rv, oper::div_assign> expression, Allocator& alloc) {
		auto right_eval = evaluator<rv>::temporary_injection(expression.right, alloc);
		return make_bin_expr<oper::div_assign>(expression.left, right_eval);

	}

	template<class expression_t, class Allocator>
	static auto evaluate_aliased(expression_t expression, Allocator& alloc) {
		return evaluator<expression_t>::temporary_injection(expression.right, alloc);
	}

	template<class expression, class Allocator>
	static void deallocate_temporaries(expression expr, Allocator& alloc) {
		evaluator<expression>::deallocate_temporaries(expr, alloc);
	}

private:
	template<class allocator, class expression_t, class Allocator>
	static auto evaluate_temporaries(expression_t expression, Allocator& alloc) {

//Use to check if a temporary is created
#ifdef BC_DISABLE_TEMPORARIES
		return expression;
#else
		return evaluator<expression_t>::template temporary_injection<allocator>(expression, alloc);
#endif
	}};

}
}
}



#endif /* PARSE_TREE_COMPLEX_EVALUATOR_H_ */
