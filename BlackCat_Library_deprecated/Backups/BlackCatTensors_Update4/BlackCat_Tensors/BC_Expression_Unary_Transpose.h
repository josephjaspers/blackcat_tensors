/*
 * BC_Expression_Unary_Transpose.h
 *
 *  Created on: Dec 2, 2017
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_UNARY_TRANSPOSE_H_
#define BC_EXPRESSION_UNARY_TRANSPOSE_H_

template<class, class, int, int>
class transpose_expression;

#include "BC_Expression_Base.h"
template<class T, class ml, int rows, int cols>
class transpose_expression : expression<transpose_expression<T, ml, rows, cols>> {
public:
	using this_type = transpose_expression<T, ml, rows, cols>;

	T* data;

	transpose_expression<T, ml, rows, cols>(T* dat) :
			data(dat) {
	}

	__attribute__((always_inline)) auto& operator [](int index) {
		return data[(int)floor(index / rows) + (index % rows) * cols];
	}
	__attribute__((always_inline)) const auto& operator[](int index) const {
		return data[(int) floor(index / rows) + (index % rows) * cols];
	}
};

#endif /* BC_EXPRESSION_UNARY_TRANSPOSE_H_ */
