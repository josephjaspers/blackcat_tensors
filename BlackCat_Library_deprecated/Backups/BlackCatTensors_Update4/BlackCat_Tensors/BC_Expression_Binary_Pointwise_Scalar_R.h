/*
 * BC_Expression_Binary_Pointwise_ScalarR.h
 *
 *  Created on: Dec 2, 2017
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_BINARY_POINTWISE_SCALAR_R_H_
#define BC_EXPRESSION_BINARY_POINTWISE_SCALAR_R_H_


#include "BC_Expression_Base.h"


template<class T, class operation, class lv, class rv>
class binary_expression_scalar_R : expression<binary_expression_scalar_R<T, operation, lv, rv>> {
public:

	using this_type = binary_expression_scalar_R<T, operation, lv, rv>;

	operation oper;

	lv left;
	rv right;

	inline __attribute__((always_inline)) binary_expression_scalar_R(lv l, rv r) :
			left(l), right(r) {
	}

	inline __attribute__((always_inline)) auto operator [](int index) const {
		return oper(left[index], right[0]);
	}
};



#endif /* BC_EXPRESSION_BINARY_POINTWISE_SCALAR_R_H_ */
