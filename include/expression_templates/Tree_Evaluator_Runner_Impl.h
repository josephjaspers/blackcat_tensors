/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PARSE_TREE_COMPLEX_EVALUATOR_H_
#define PARSE_TREE_COMPLEX_EVALUATOR_H_

#include "Tree_Evaluator_Common.h"
#include "Tree_Evaluator_Array.h"
#include "Tree_Evaluator_Temporary.h"
#include "Tree_Evaluator_Binary_Linear.h"
#include "Tree_Evaluator_Binary_NonLinear.h"
#include "Tree_Evaluator_Unary.h"

#include "Tree_Evaluator_BLAS.h"

namespace BC{
namespace et     {
namespace tree {

struct Greedy_Evaluator {

	template<class lv, class rv> __BChot__
	static  auto evaluate(Binary_Expression<lv, rv, oper::add_assign> expression) {
		auto right = evaluator<rv>::linear_evaluation(expression.right, injector<lv, 1, 1>(expression.left));
		return make_bin_expr<oper::add_assign>(expression.left, evaluate_temporaries(right));
	}

	template<class lv, class rv> __BChot__
	static auto evaluate(Binary_Expression<lv, rv, oper::sub_assign> expression) {
		auto right = evaluator<rv>::linear_evaluation(expression.right, injector<lv, -1, 1>(expression.left));
		return make_bin_expr<oper::add_assign>(expression.left, evaluate_temporaries(right));
	}

	template<class lv, class rv> __BChot__
	static auto evaluate(Binary_Expression<lv, rv, oper::assign> expression) {
		auto right = evaluator<rv>::injection(expression.right, injector<lv, 1, 0>(expression.left));
		return make_bin_expr<oper::assign>(expression.left, evaluate_temporaries(right));

	}

	template<class lv, class rv> __BChot__
	static auto evaluate(Binary_Expression<lv, rv, oper::mul_assign> expression) {
		auto right_eval =  evaluator<rv>::temporary_injection(expression.right);
		return make_bin_expr<oper::mul_assign>(expression.left, right_eval);

	}

	template<class lv, class rv>__BChot__
	static auto evaluate(Binary_Expression<lv, rv, oper::div_assign> expression) {
		auto right_eval = evaluator<rv>::temporary_injection(expression.right);
		return make_bin_expr<oper::div_assign>(expression.left, right_eval);

	}

	template<class expression_t>__BChot__
	static auto evaluate_aliased(expression_t expression) {
		return evaluator<expression_t>::temporary_injection(expression.right);
	}

	template<class expression> __BChot__
	static void deallocate_temporaries(expression expr) {
		evaluator<expression>::deallocate_temporaries(expr);
	}

private:
	template<class expression_t>__BChot__
	static auto evaluate_temporaries(expression_t expression) {

//Use to check if a temporary is created
#ifdef BC_DISABLE_TEMPORARIES
		return expression;
#else
		return evaluator<expression_t>::temporary_injection(expression);
#endif
	}};

}
}
}



#endif /* PARSE_TREE_COMPLEX_EVALUATOR_H_ */
