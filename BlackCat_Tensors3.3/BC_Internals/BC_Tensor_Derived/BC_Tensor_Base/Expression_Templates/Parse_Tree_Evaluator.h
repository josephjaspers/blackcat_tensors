/*
 * Trivial_Evaluator.h
 *
 *  Created on: Jun 7, 2018
 *      Author: joseph
 */

#ifndef TRIVIAL_EVALUATOR_H_
#define TRIVIAL_EVALUATOR_H_

#include "Parse_Tree_Evaluator_Impl.h"

namespace BC {

template<class> class Tensor_Base;
template<class expression> //get the type of the rv_expression
using rv_of = std::decay_t<decltype(std::declval<std::decay_t<expression>>().right)>;

template<class T>
static constexpr bool INJECTION() {
	//non-trivial is true even when it is trivial
	return internal::tree::expression_tree_evaluator<std::decay_t<T>>::non_trivial_blas_injection;
}



template<class mathlib_type>
struct Evaluator {

template<class assignment, class expression>
static std::enable_if_t<!INJECTION<expression>()>
evaluate(const assignment& assign, const expression& expr) {
	static constexpr int iterator_dimension = expression::ITERATOR();
		mathlib_type::template dimension<iterator_dimension>::eval(expr);
}

template<class assignment, class expression>
static std::enable_if_t<INJECTION<expression>()>
evaluate(assignment& assign, const expression& expr) {
	using internal_t = expression;			//internal expression type (to be injected)
	using injection_t = assignment;	//the injection type

	static constexpr int iterator_dimension = expression::ITERATOR();	//the iterator for the evaluation of post inject_t

	auto post_inject_tensor = internal::tree::evaluate(expr);		//evaluate the internal tensor_type
	if (!std::is_same<injection_t, rv_of<decltype(post_inject_tensor)>>::value || BC::internal::isArray<decltype(post_inject_tensor)>()) {
		mathlib_type::template dimension<iterator_dimension>::eval(post_inject_tensor);
	}
	//destroy any temporaries made by the tree
	internal::tree::destroy_temporaries(post_inject_tensor);
}

template<class assignment, class expression>
static std::enable_if_t<INJECTION<expression>()>
evaluate_aliased(assignment& assign, const expression& expr) {
	using internal_t = expression;			//internal expression type (to be injected)
	using injection_t = assignment;	//the injection type

	static constexpr int iterator_dimension = expression::ITERATOR();	//the iterator for the evaluation of post inject_t

	auto post_inject_tensor = internal::tree::substitution_evaluate(expr);		//evaluate the internal tensor_type
	if (!std::is_same<injection_t, rv_of<decltype(post_inject_tensor)>>::value || BC::internal::isArray<decltype(post_inject_tensor)>()) {
		mathlib_type::template dimension<iterator_dimension>::eval(post_inject_tensor);
	}
	//destroy any temporaries made by the tree
	internal::tree::destroy_temporaries(post_inject_tensor);
}
template<class assignment, class expression>
static std::enable_if_t<!INJECTION<expression>()>
evaluate_aliased(assignment& assign, const expression& expr) {
	static constexpr int iterator_dimension = expression::ITERATOR();
	mathlib_type::template dimension<iterator_dimension>::eval(expr);
}

};

template<class mathlib>
struct branched {
	template<class branch> using sub_t 	= BC::internal::Array<branch::DIMS(), _scalar<branch>, mathlib>;
	template<class branch> using eval_t = BC::internal::binary_expression<sub_t<branch>, branch, BC::oper::assign>;

	template<class branch>
	static std::enable_if_t<BC::internal::isArray<std::decay_t<branch>>(), const branch&> evaluate(const branch& expression) { return expression; }

	template<class branch>
	static std::enable_if_t<!BC::internal::isArray<std::decay_t<branch>>(), sub_t<std::decay_t<branch>>> evaluate(const branch& expression)
	{
		sub_t<std::decay_t<branch>> cached_branch(expression.inner_shape());
		eval_t<std::decay_t<branch>> assign_to_expression(cached_branch, expression); //create an expression to assign to the left_cached
		Evaluator<mathlib>::evaluate(cached_branch, assign_to_expression);
		return cached_branch;
	}

};
}



#endif /* TRIVIAL_EVALUATOR_H_ */
