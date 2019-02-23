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
namespace expression_templates {
namespace tree {
namespace detail {

template<class expression_t, class voider=void>
struct temporary_evaluator;

template<class expression_t>
struct temporary_evaluator<expression_t, std::enable_if_t<!optimizer<expression_t>::requires_greedy_eval>> {
	template<class Allocator>
	static auto evaluate_temporaries(expression_t expression, Allocator& alloc) {
		return expression;
	}
};


template<class expression_t>
struct temporary_evaluator<expression_t, std::enable_if_t<optimizer<expression_t>::requires_greedy_eval>> {
	template<class Allocator>
	static auto evaluate_temporaries(expression_t expression, Allocator& alloc) {
		auto expr = optimizer<expression_t>::template temporary_injection(expression, alloc);
		return temporary_evaluator<std::decay_t<decltype(expr)>>::evaluate_temporaries(expr, alloc);
	}
};

}

struct Greedy_Evaluator {

	template<class lv, class rv, class Allocator>
	static  auto evaluate(Binary_Expression<lv, rv, oper::add_assign> expression, Allocator& alloc) {
		auto right = optimizer<rv>::linear_evaluation(expression.right, injector<lv, 1, 1>(expression.left), alloc);
		return make_bin_expr<oper::add_assign>(expression.left, evaluate_temporaries(right, alloc));
	}

	template<class lv, class rv, class Allocator>
	static auto evaluate(Binary_Expression<lv, rv, oper::sub_assign> expression, Allocator& alloc) {
		auto right = optimizer<rv>::linear_evaluation(expression.right, injector<lv, -1, 1>(expression.left), alloc);
		return make_bin_expr<oper::sub_assign>(expression.left, evaluate_temporaries(right, alloc));
	}

	template<class lv, class rv, class Allocator, class=std::enable_if_t<!optimizer<rv>::partial_blas_expr>>
	static auto evaluate(Binary_Expression<lv, rv, oper::assign> expression, Allocator& alloc) {
		auto right = optimizer<rv>::injection(expression.right, injector<lv, 1, 0>(expression.left), alloc);
		return make_bin_expr<oper::assign>(expression.left, evaluate_temporaries(right, alloc));
	}

	template<class lv, class rv, class Allocator, class=std::enable_if_t<optimizer<rv>::partial_blas_expr>, int=0>
	static auto evaluate(Binary_Expression<lv, rv, oper::assign> expression, Allocator& alloc) {
		auto right = optimizer<rv>::linear_evaluation(expression.right, injector<lv, 1, 0>(expression.left), alloc);
		return make_bin_expr<oper::add_assign>(expression.left, evaluate_temporaries(right, alloc));
	}


	template<
		class lv,
		class rv,
		class Allocator,
		class assignment_oper,
		class=std::enable_if_t<BC::oper::operation_traits<assignment_oper>::is_assignment_operation>
	>
	static auto evaluate(Binary_Expression<lv, rv, assignment_oper> expression, Allocator& alloc) {
		auto right_eval =  optimizer<rv>::temporary_injection(expression.right,  alloc);
		return make_bin_expr<assignment_oper>(expression.left, right_eval);

	}

	template<class lv, class rv, class op, class Allocator>
	static auto evaluate_aliased(Binary_Expression<lv, rv, op> expression, Allocator& alloc) {
		auto right = optimizer<rv>::temporary_injection(expression.right, alloc);
		return make_bin_expr<op>(expression.left, right);
	}

	template<class expression, class Allocator>
	static void deallocate_temporaries(expression expr, Allocator& alloc) {
		optimizer<expression>::deallocate_temporaries(expr, alloc);
	}

private:

	template<class expression_t, class Allocator>
	static auto evaluate_temporaries(expression_t expression, Allocator& alloc) {
		return detail::temporary_evaluator<expression_t>::evaluate_temporaries(expression, alloc);

	}

};

}
}
}



#endif /* PARSE_TREE_COMPLEX_EVALUATOR_H_ */
