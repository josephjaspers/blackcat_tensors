/*
 * Trivial_Evaluator.h
 *
 *  Created on: Jun 7, 2018
 *      Author: joseph
 */

#ifndef TRIVIAL_EVALUATOR_H_
#define TRIVIAL_EVALUATOR_H_

#include "Tree_Evaluator_Runner_Impl.h"

namespace BC {
namespace internal {
template<class mathlib_type>
struct Lazy_Evaluator {


//------------------------------------------------Purely lazy evaluation----------------------------------//
template< class expression>
static std::enable_if_t<!INJECTION<expression>()>
evaluate(const expression& expr) {
	static constexpr int iterator_dimension = expression::ITERATOR();
	mathlib_type::template dimension<iterator_dimension>::eval(expr);
}
//------------------------------------------------Purely lazy alias evaluation----------------------------------//
template< class expression>
static std::enable_if_t<!INJECTION<expression>()>
evaluate_aliased(const expression& expr) {
	static constexpr int iterator_dimension = expression::ITERATOR();
	mathlib_type::template dimension<iterator_dimension>::eval(expr);
}
//------------------------------------------------Greedy evaluation (BLAS function call detected)----------------------------------//
template< class expression>
static std::enable_if_t<INJECTION<expression>()>
evaluate(const expression& expr) {
	auto greedy_evaluated_expr = internal::tree::Greedy_Evaluator::evaluate(expr);

	if (is_expr<decltype(greedy_evaluated_expr)>()) {
		static constexpr int iterator_dimension = expression::ITERATOR();
		mathlib_type::template dimension<iterator_dimension>::eval(greedy_evaluated_expr);
	}
	//destroy any temporaries made by the tree
	internal::tree::Greedy_Evaluator::destroy_temporaries(greedy_evaluated_expr);
}
//------------------------------------------------Greedy evaluation (BLAS function call detected), skip injection optimization--------------------//
template< class expression>
static std::enable_if_t<INJECTION<expression>()>
evaluate_aliased(const expression& expr) {
	static constexpr int iterator_dimension = expression::ITERATOR();	//the iterator for the evaluation of post inject_t

	auto greedy_evaluated_expr = internal::tree::Greedy_Evaluator::substitution_evaluate(expr);		//evaluate the internal tensor_type
	if (is_expr<decltype(greedy_evaluated_expr)>()) {
		mathlib_type::template dimension<iterator_dimension>::eval(greedy_evaluated_expr);
	}
	//destroy any temporaries made by the tree
	internal::tree::Greedy_Evaluator::destroy_temporaries(greedy_evaluated_expr);
}
};

template<class mathlib>
struct CacheEvaluator {
	template<class branch> using sub_t 	= BC::internal::Array<branch::DIMS(), BC::internal::scalar_of<branch>, mathlib>;
	template<class branch> using eval_t = BC::internal::binary_expression<sub_t<branch>, branch, BC::internal::oper::assign>;

	template<class branch> //The branch is an array, no evaluation required
	static std::enable_if_t<BC::is_array<std::decay_t<branch>>(), const branch&> evaluate(const branch& expression) { return expression; }

	template<class branch> //Create and return an array_core created from the expression
	static std::enable_if_t<!BC::is_array<std::decay_t<branch>>(), sub_t<std::decay_t<branch>>> evaluate(const branch& expression)
	{
		sub_t<std::decay_t<branch>> cached_branch(expression.inner_shape());
		eval_t<std::decay_t<branch>> assign_to_expression(cached_branch, expression);
		Lazy_Evaluator<mathlib>::evaluate(assign_to_expression);
		return cached_branch;
	}

};

template<class array_t, class expression_t>
void evaluate_to(array_t array, expression_t expr) {
	static_assert(is_array<array_t>(), "MAY ONLY EVALUATE TO ARRAYS");
	Lazy_Evaluator<internal::mathlib_of<array_t>>::evaluate(internal::binary_expression<array_t, expression_t, internal::oper::assign>(array, expr));
}

template<class expression_t>
void evaluate(expression_t expr) {
	Lazy_Evaluator<mathlib_of<expression_t>>::evaluate(expr);
}


}
}


#endif /* TRIVIAL_EVALUATOR_H_ */
