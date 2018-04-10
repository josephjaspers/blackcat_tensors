/*
 * BC_Expression_Binary.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_BINARY_POINTWISE_SAME_H_
#define BC_EXPRESSION_BINARY_POINTWISE_SAME_H_

#include "BC_Expression_Base.h"

template<class T, class operation, class lv, class rv>
class binary_expression : expression<binary_expression<T, operation, lv, rv>> {
public:

	using this_type = binary_expression<T, operation, lv, rv>;

	operation oper;

	lv left;
	rv right;

	inline __attribute__((always_inline)) binary_expression(lv l, rv r) :
			left(l), right(r) {
	}
	inline __attribute__((always_inline))
	auto operator [](int index) const {
		return oper(left[index], right[index]);
	}
};

#endif /* BC_EXPRESSION_BINARY_POINTWISE_SAME_H_ */
