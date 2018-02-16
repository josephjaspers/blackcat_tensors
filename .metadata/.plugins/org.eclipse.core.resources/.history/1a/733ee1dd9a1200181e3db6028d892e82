/*
 * Transpose_Wrapper.h
 *
 *  Created on: Dec 19, 2017
 *      Author: joseph
 */
#ifdef  __CUDACC__
#ifndef EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_
#define EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_
#include <iostream>
#include <cmath>
#include "Expression_Base.cu"
#include "../BC_MetaTemplateFunctions/Adhoc.h"
#include "../BlackCat_Internal_Definitions.h"

namespace BC {
template<class T, class functor_type>
struct unary_expression_transpose : expression<T, unary_expression_transpose<T, functor_type>>
{

	functor_type array;
	const bool vector = false;
	const int rows;
	const int cols;
	const int LD;

	unary_expression_transpose(functor_type p, int rows, int cols, int ld)
	: array(p), rows(rows), cols(cols), LD(ld), vector(cols == 1 || rows == 1) {}



private:
	template<class U> __BC_gcpu__
	static int bc_floor(U number) {
		return ((int)number) > number ? (int)(number - .5) : number; //if casting the number to an integer is greater (if it rounded up) subtract .5 and the round again
	}

public:


	__attribute__((always_inline)) __BC_gcpu__  auto operator [](int index) -> decltype(array[index]) {


		if (vector)
			return array[index];
		else
			return array[(int)(index / cols) + (index % cols) * LD];
	}
	__attribute__((always_inline))  __BC_gcpu__ auto operator[](int index) const  -> const decltype(array[index])  {

		if (vector)
			return array[index];
		else
			return array[(int)(index / cols) + (index % cols) * LD];
	}
};
}
#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
#endif
