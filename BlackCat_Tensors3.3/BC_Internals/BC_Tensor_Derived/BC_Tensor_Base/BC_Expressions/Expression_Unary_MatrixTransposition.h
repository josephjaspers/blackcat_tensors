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

	unary_expression_transpose(functor_type p) : array(p) {}

	__BCinline__ auto innerShape() const { return l_array([=](int i) { return i == 0 ? array.cols() : i == 1 ? array.rows() : 1; }); }
	__BCinline__ const auto outerShape() const { return array.outerShape(); }

	__BCinline__ auto operator [](int index) -> decltype(array[index]) {
			return array[DIMS() == 1 ? index : (int)(index / this->rows()) + (index % this->rows()) * this->LD_rows()];
	}
	__BCinline__ auto operator[](int index) const  -> const decltype(array[index])  {
			return array[DIMS() == 1 ? index : (int)(index / this->rows()) + (index % this->rows()) * this->LD_rows()];
	}
};
}
#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
