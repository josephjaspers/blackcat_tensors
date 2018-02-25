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
#include "BlackCat_Internal_Definitions.h"
#include <vector>

namespace BC {
template<class T, class functor_type>
struct unary_expression_transpose : expression<T, unary_expression_transpose<T, functor_type>>
{
	const functor_type& array;
	const bool vector = rows() == 1 || cols() == 1;

	unary_expression_transpose(const functor_type& p) : array(p) {}

	__BC_gcpu__	int rank() const { return array.rank(); }
	__BC_gcpu__ int rows() const { return array.cols(); }
	__BC_gcpu__ int cols() const { return array.rows(); }
	__BC_gcpu__ int size() const { return array.size(); }
	__BC_gcpu__ int LD_rows() const { return array.LD_rows(); }
	__BC_gcpu__ int LD_cols() const { return array.LD_cols(); }
	__BC_gcpu__ int dimension(int i)	const { return array.dimension(i); }
//	__BC_gcpu__ const std::vector<int> InnerShape() const { array.InnerShape(); } //return std::vector<int> { rows(), cols() }; }
//	__BC_gcpu__ const auto OuterShape() const { return array.OuterShape(); }

	void printDimensions() const {
		std::cout << "[" << rows() << "]" << "[" << cols() << "]" << std::endl;
	}
	void printLDDimensions()	const { array.printLDDimensions(); }


	__attribute__((always_inline)) __BC_gcpu__  auto operator [](int index) -> decltype(array[index]) {
		if (vector)
			return array[index];
		else
			return array[(int)(index / rows()) + (index % rows()) * LD_rows()];
	}
	__attribute__((always_inline))  __BC_gcpu__ auto operator[](int index) const  -> const decltype(array[index])  {
		if (vector)
			return array[index];
		else
			return array[(int)(index / rows()) + (index % rows()) * LD_rows()];
	}
};
}
#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
#endif
