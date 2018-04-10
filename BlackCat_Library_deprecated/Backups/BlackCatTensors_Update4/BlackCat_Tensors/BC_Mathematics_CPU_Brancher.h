/*
 * BC_Mathematics_CPU_Brancher.h
 *
 *  Created on: Dec 2, 2017
 *      Author: joseph
 */

#ifndef BC_MATHEMATICS_CPU_BRANCHER_H_
#define BC_MATHEMATICS_CPU_BRANCHER_H_

#include "BC_Expression_Binary_Pointwise_Same.h"

namespace BC_Expression_Brancher {

	template<class ... operations>
	struct oper_stack;

	template<class ... twigs>
	struct twig_stack;

	template<class, class >
	struct combine;

	template<template<class ...> class T, class... expr1, class... expr2>
	struct combine< T<expr1>, T<expr2> > {
		using type = T<expr1, expr2>;
	};

	template<class addTo, class ... from>
	class insert_front;

	template<template<class ...> class add, class... to, class... from>
	struct insert_front<add<to...>, from...> {
		using type = add<from..., to...>;
	};

	template<class T, class expression, class oper_stack, class twig_stack>
	struct gatherBranches;

	template<class T, class array_type, class ... opers, class ... twigs>
	struct gatherBranches<T, array_type, oper_stack<opers...>, twig_stack<twigs...>> {
		using operation_stack = typename oper_stack<opers...>;
		using exression_stack = typename twig_stack<twigs..., array_type>;
	};

	template<class T, class U, class oper, class lv, class rv, class ... opers, class ... twigs>
	struct gatherBranches<T, binary_expression<U, oper, lv, rv>, oper_stack<opers...>, twig_stack<twigs...>> {

		/*
		 * Recursively generates the operation and expression stack. -- This is a compile time parse tree
		 * IE y = a + b * d - c;
		 *
		 * Creates expression(b,d)
		 * expression(a, expression(b,d))
		 * expression(expression(a, expression(b,d)), c)
		 *
		 * Results operation stack ->  (*) (+) (-)
		 * Expression Stack        ->  BD, A(BD), (A(BD))C,
		 */

		using operation_stack_left = typename gatherBranches<T, lv, oper_stack<opers...>, twig_stack<twigs...>::operation_stack_left;
		using exression_stack_left = typename gatherBranches<T, lv, oper_stack<opers...>, twig_stack<twigs...>::exression_stack_left;

		using operation_stack_right = typename gatherBranches<T, rv, oper_stack<opers...>, twig_stack<twigs...>::operation_stack_left;
		using exression_stack_right = typename gatherBranches<T, rv, oper_stack<opers...>, twig_stack<twigs...>::exression_stack_left;

		using combined_operation_stack = typename combine<operation_stack_left, operation_stack_right>::type;
		using combined_expression_stack = typename combine<exression_stack_left, exression_stack_right>::type;

		using operation_stack = typename insert_front<combined_operation_stack, oper>::type;
		using expression_stack = typename insert_front<combined_expression_stack, lv, rv>::type;
	};

#include "BC_Expression_Binary_Functors.h"

	template<class assign, class oper_stack, class expression_stack>
	struct evaluator;

	template<class assign, class ... opers, class front_oper, class ... twigs, class l_twig, class r_twig>
	struct evaluator<assign, oper_stack<front_oper, opers...>, twig_stack<l_twig, r_twig, twigs...>>;

	template<class assign, class ... opers, class front_oper, class ... twigs, class l_twig, class r_twig>
	struct evaluator<assign, oper_stack<BC::add, opers...>, twig_stack<l_twig, r_twig, twigs...>> {
	};

	template<class assign, class ... opers, class front_oper, class ... twigs, class l_twig, class r_twig>
	struct evaluator<assign, oper_stack<BC::sub, opers...>, twig_stack<l_twig, r_twig, twigs...>> ;

	template<class assign, class ... opers, class front_oper, class ... twigs, class l_twig, class r_twig>
	struct evaluator<assign, oper_stack<BC::mul, opers...>, twig_stack<l_twig, r_twig, twigs...>> ;

	template<class assign, class ... opers, class front_oper, class ... twigs, class l_twig, class r_twig>
	struct evaluator<assign, oper_stack<BC::div, opers...>, twig_stack<l_twig, r_twig, twigs...>> ;

	template<class assign>
	struct evaluator<assign, oper_stack<>, twig_stack<>> ;
}

#endif /* BC_MATHEMATICS_CPU_BRANCHER_H_ */
