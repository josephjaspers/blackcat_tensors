/*
 * BC_Expression_Unary.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_UNARY_H_
#define BC_EXPRESSION_UNARY_H_

#include "BC_Expression_Base.h"


template<class T, class operation, class value>
struct unary_expression : expression<unary_expression<T, operation, value>> {
public:

	using this_type = unary_expression<T, operation, value>;

	operation oper;
	value val;

	inline __attribute__((always_inline)) unary_expression(value v) :
			val(v) {
	}

	inline __attribute__((always_inline)) auto operator [](int index) const {
		return oper(val[index]);
	}
};

#endif /* BC_EXPRESSION_UNARY_H_ */
