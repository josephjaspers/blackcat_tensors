/*
 * Parse_Tree_Complex_Evaluator.h
 *
 *  Created on: Jun 18, 018
 *      Author: joseph
 */

#ifndef PARSE_TREE_COMPLEX_EVALUATOR_H_
#define PARSE_TREE_COMPLEX_EVALUATOR_H_

#include "Operations/Binary.h"
#include "Operations/Unary.h"
#include "BC_Utility/Temporary.h"
#include "Tree_Evaluator_Array.h"
#include "Tree_Evaluator_Binary_Linear.h"
#include "Tree_Evaluator_Binary_NonLinear.h"
#include "Tree_Evaluator_Unary.h"
#include "Tree_Evaluator_BLAS.h"
#include "Tree_Evaluator_Temporary.h"

namespace BC{
namespace internal {
namespace tree {

template<class lv, class rv, class op>
auto substitution_evaluate(binary_expression<lv, rv, op> expression);

struct sub_eval_recursion {
	template<class lv, class rv, class op>
	static auto function(binary_expression<lv, rv, op> expression) {
		return substitution_evaluate(evaluator<binary_expression<lv, rv, op>>::replacement(expression));
	}
};
struct sub_eval_terminate {
	template<class lv, class rv, class op>
	static auto function(binary_expression<lv, rv, op> expression) {
		return expression;
	}
};

template<class lv, class rv, class op>
auto substitution_evaluate(binary_expression<lv, rv, op> expression) {
	using impl = std::conditional_t<evaluator<binary_expression<lv, rv, op>>::non_trivial_blas_injection,
					sub_eval_recursion, sub_eval_terminate>;

	return impl::function(expression);
}

template<class expression>
void destroy_temporaries(expression expr) {
	evaluator<expression>::destroy_temporaries(expr);
}

template<class lv, class rv>
auto evaluate(binary_expression<lv, rv, oper::add_assign> expression) {
	auto right = evaluator<rv>::linear_evaluation(expression.right, injector<lv, 1, 1>(expression.left));
	return substitution_evaluate(binary_expression<lv, std::decay_t<decltype(right)>, oper::add_assign>(expression.left, right));
}
template<class lv, class rv>
auto evaluate(binary_expression<lv, rv, oper::sub_assign> expression) {
	auto right = evaluator<rv>::linear_evaluation(expression.right, injector<lv, -1, 1>(expression.left));
	return substitution_evaluate(binary_expression<lv, std::decay_t<decltype(right)>, oper::sub_assign>(expression.left, right));
}
template<class lv, class rv>
auto evaluate(binary_expression<lv, rv, oper::assign> expression) {
	auto right = evaluator<rv>::injection(expression.right, injector<lv, 1, 0>(expression.left));
	return substitution_evaluate(binary_expression<lv, std::decay_t<decltype(right)>, oper::assign>(expression.left, right));
}
template<class lv, class rv>
auto evaluate(binary_expression<lv, rv, oper::mul_assign> expression) {
	return substitution_evaluate(expression);
}
template<class lv, class rv>
auto evaluate(binary_expression<lv, rv, oper::div_assign> expression) {
	return substitution_evaluate(expression);
}




}
}
}



#endif /* PARSE_TREE_COMPLEX_EVALUATOR_H_ */
