/*
 * BC_Expression.h
 *
 *  Created on: Nov 20, 2017
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_BINARY_H_
#define BC_EXPRESSION_BINARY_H_

#include "BC_Expression_Functors.h"
#include <functional>
#include "BC_Tensor_Super_Queen.h"



template<class operation, class ml, class lv, class rv, int ... dimensions>
class binary_expression : public Tensor_Queen<binary_expression<operation, ml, lv, rv, dimensions...>, ml, dimensions...> {
public:

	using this_type = binary_expression<operation,ml, lv, rv, dimensions...>;

	operation oper;

	lv left;
	rv right;

	inline __attribute__((always_inline)) binary_expression(lv l, rv r) :
			left(l), right(r) {
	}

	inline __attribute__((always_inline)) auto operator [](int index) const {
		return oper(left[index], right[index]);
	}
};

template<class, class, bool>
class Scalar;

//scalar left
template<class operation, class ml, class T, class rv, bool isID, int ... dimensions>
class binary_expression<operation, ml, Scalar<T, ml, isID>, rv, dimensions...> : public Tensor_Queen<binary_expression<operation, ml, Scalar<T, ml, isID>, rv, dimensions...>,
		ml, dimensions...> {
public:

	using this_type = binary_expression<operation, ml, Scalar<T,ml, isID>, rv, dimensions...>;
	using lv = Scalar<T, ml, isID>;
	operation oper;

	lv left;
	rv right;

	inline __attribute__((always_inline)) binary_expression(lv l, rv r) :
			left(l), right(r) {
	}

	inline __attribute__((always_inline)) auto operator [](int index) const {
		return oper(left.getData(), right[index]);
	}
};
//Scalar right
template<class operation, class ml, class T, class lv, bool isID, int ... dimensions>
class binary_expression<operation, ml, lv, Scalar<T, ml, isID>, dimensions...> : public Tensor_Queen<binary_expression<operation, ml, lv, Scalar<T, ml, isID>, dimensions...>,
		ml, dimensions...> {
public:

	using this_type = binary_expression<operation, ml, lv, Scalar<T,ml, isID>, dimensions...>;
	using rv = Scalar<T, ml, isID>;
	operation oper;

	lv left;
	rv right;

	inline __attribute__((always_inline)) binary_expression(lv l, rv r) :
			left(l), right(r) {
	}

	inline __attribute__((always_inline)) auto operator [](int index) const {
		return oper(left[index], right.getData());
	}
};

#endif /* BC_EXPRESSION_BINARY_H_ */
