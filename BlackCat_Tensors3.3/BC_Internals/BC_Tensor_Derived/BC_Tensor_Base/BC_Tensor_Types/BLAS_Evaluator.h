/*
 * Trivial_Evaluator.h
 *
 *  Created on: Jun 7, 2018
 *      Author: joseph
 */

#ifndef TRIVIAL_EVALUATOR_H_
#define TRIVIAL_EVALUATOR_H_

#include "BLAS_Injection_Wrapper.h"
#include "Core_Substitution.h"
#include <cxxabi.h>

namespace BC {


std::string removeNS( const std::string & source, const std::string & namespace_ )
{
    std::string dst = source;
    size_t position = source.find( namespace_ );
    while ( position != std::string::npos )
    {
        dst.erase( position, namespace_.length() );
        position = dst.find( namespace_, position + 1 );
    }
    return dst;
}

template<class T>
std::string type_name() {
	int status;
	  std::string demangled = abi::__cxa_demangle(typeid(T).name(),0,0,&status);
	  return removeNS(removeNS(removeNS(demangled, "BC::"), "internal::"), "oper::");
}



template<class> class Tensor_Base;

template<class mathlib_type, bool BARRIER>
struct Evaluator {

template<class assignment, class expression> __BC_host_inline__
static std::enable_if_t<!BC::internal::INJECTION<expression>()>
evaluate(const assignment& assign, const expression& expr) {
	static constexpr int iterator_dimension = expression::ITERATOR();
	if (BARRIER)
		mathlib_type::template dimension<iterator_dimension>::eval(expr);
	else
		mathlib_type::template dimension<iterator_dimension>::eval_unsafe(expr);
}

template<class expression> //get the type of the rv_expression
using rv_of = std::decay_t<decltype(std::declval<expression>().right)>;
//
template<class assignment, class expression> __BC_host_inline__
static std::enable_if_t<BC::internal::INJECTION<expression>()>
evaluate(assignment& assign, const expression& expr) {
	using internal_t = expression;			//internal expression type (to be injected)
	using injection_t = assignment;	//the injection type
//	using tree					  = BC::internal::traversal<expression>;
//	using rotated_expression_tree = typename BC::internal::traversal<expression>::type;	//the conversion type after injection
	using rotated_expression_tree = typename expression::injection_type;

//	std::cout << type_name<rotated_expression_tree> () << std::endl;
//	std::co

	static constexpr int iterator_dimension = rotated_expression_tree::ITERATOR();	//the iterator for the evaluation of post inject_t
	static constexpr int alpha_mod = expression::alpha_mod();
	static constexpr int beta_mod = expression::beta_mod();

	auto post_inject_tensor = rotated_expression_tree(expr, BC::internal::injection_wrapper<assignment, alpha_mod, beta_mod>(assign));		//evaluate the internal tensor_type

	if (!std::is_same<injection_t, rv_of<rotated_expression_tree>>::value) {
	if (BARRIER)
		mathlib_type::template dimension<iterator_dimension>::eval(post_inject_tensor);
	else
		mathlib_type::template dimension<iterator_dimension>::eval_unsafe(post_inject_tensor);
	} else {
	}
}


//template<class assignment, class expression> __BC_host_inline__
//static std::enable_if_t<BC::internal::INJECTION<expression>()>
//evaluate(assignment& assign, const expression& expr) {
//	using internal_t = expression;			//internal expression type (to be injected)
//	using injection_t = assignment;	//the injection type
////	using tree					  = BC::internal::traversal<expression>;
//	using rotated_expression_tree = typename expression::injection_type;
//
//	static constexpr int iterator_dimension = rotated_expression_tree::ITERATOR();	//the iterator for the evaluation of post inject_t
//	static constexpr int alpha_mod = expression::alpha_mod();
//	static constexpr int beta_mod = expression::beta_mod();
//
//	auto post_inject_tensor = rotated_expression_tree(expr, BC::internal::injection_wrapper<assignment, alpha_mod, beta_mod>(assign));		//evaluate the internal tensor_type
//
//	if (!std::is_same<injection_t, rv_of<rotated_expression_tree>>::value) {
//	if (BARRIER)
//		mathlib_type::template dimension<iterator_dimension>::eval(post_inject_tensor);
//	else
//		mathlib_type::template dimension<iterator_dimension>::eval_unsafe(post_inject_tensor);
//	} else {
//	}
//}
};

template<class mathlib, bool BARRIER>
struct branched {
	template<class branch> using sub_t = BC::internal::Tensor_Substitution<tensor_of_t<branch::DIMS(), _scalar<branch>, mathlib>>;
	template<class branch> using eval_t =BC::internal::binary_expression<sub_t<branch>, branch, BC::oper::assign>;


	template<class branch> __BC_host_inline__
	static std::enable_if_t<BC::internal::isCore<std::decay_t<branch>>(), const branch&> evaluate(const branch& expression) { return expression; }

	template<class branch> __BC_host_inline__
	std::enable_if_t<!BC::internal::isCore<std::decay_t<branch>>(), sub_t<std::decay_t<branch>>> evaluate(const branch& expression)
	{
		sub_t<std::decay_t<branch>> cached_branch(expression.inner_shape());
		eval_t<std::decay_t<branch>> assign_to_expression(cached_branch, expression); //create an expression to assign to the left_cached
		Evaluator<mathlib, true>::evaluate(cached_branch, assign_to_expression);
		return cached_branch;
	}

};
}



#endif /* TRIVIAL_EVALUATOR_H_ */
