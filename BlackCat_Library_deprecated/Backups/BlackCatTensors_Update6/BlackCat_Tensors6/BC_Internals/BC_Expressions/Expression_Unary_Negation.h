/*
 * Expression_Unary_Negation.h
 *
 *  Created on: Dec 26, 2017
 *      Author: joseph
 */

#ifndef EXPRESSION_UNARY_NEGATION_H_
#define EXPRESSION_UNARY_NEGATION_H_

#include "Expression_Base.h"
#include "../BlackCat_Internal_GlobalUnifier.h"
#include "../BC_MetaTemplateFunctions/Typeclass_FunctionType.h"

template<class T, class functor_type>
struct unaryExpression_negation {

	functor_type array;
	unaryExpression_negation( 	   functor_type cpy) : array(cpy) {}
	unaryExpression_negation(const functor_type& cpy) : array(cpy) {}
	unaryExpression_negation(const functor_type&& cpy) : array(cpy) {}

	T operator [] (int index) const {
		return - array[index];
	}
};


#endif /* EXPRESSION_UNARY_NEGATION_H_ */
