/*
 * Transpose_Wrapper.h
 *
 *  Created on: Dec 19, 2017
 *      Author: joseph
 */
#ifndef EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_
#define EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_
#include "Expression_Base.h"
#include <vector>

namespace BC {
template<class T, class functor_type>
struct unary_expression_transpose : expression<T, unary_expression_transpose<T, functor_type>>
{
	functor_type array;
	__BCinline__ static constexpr int DIMS() { return functor_type::DIMS(); }

	const bool vector = rows() == 1 || cols() == 1;



	unary_expression_transpose(functor_type p) : array(p) {}

	__BCinline__	int dims() const { return array.dims(); }
	__BCinline__ int rows() const { return array.cols(); }
	__BCinline__ int cols() const { return array.rows(); }
	__BCinline__ int size() const { return array.size(); }
	__BCinline__ int LD_rows() const { return array.LD_rows(); }
	__BCinline__ int LD_cols() const { return array.LD_cols(); }
	__BCinline__ int dimension(int i)	const { return array.dimension(i); }
	__BCinline__ auto innerShape() const { return array.innerShape(); }
	__BCinline__ const auto outerShape() const { return array.outerShape(); }

	void printDimensions() const {
		std::cout << "[" << rows() << "]" << "[" << cols() << "]" << std::endl;
	}
	void printLDDimensions()	const { array.printLDDimensions(); }


	__BCinline__ auto operator [](int index) -> decltype(array[index]) {
		if (vector)
			return array[index];
		else
			return array[(int)(index / rows()) + (index % rows()) * LD_rows()];
	}
	__BCinline__ auto operator[](int index) const  -> const decltype(array[index])  {
		if (vector)
			return array[index];
		else
			return array[(int)(index / rows()) + (index % rows()) * LD_rows()];
	}
};
}
#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
