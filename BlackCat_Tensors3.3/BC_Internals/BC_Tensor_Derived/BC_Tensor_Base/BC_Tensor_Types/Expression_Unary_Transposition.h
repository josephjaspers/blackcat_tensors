/*
 * Transpose_Wrapper.h
 *
 *  Created on: Dec 19, 2017
 *      Author: joseph
 */
#ifndef EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_
#define EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_
#include "Expression_Base.h"
#include "Expression_Binary_Functors.h"
#include <vector>

namespace BC {
class transpose;

template<class functor_type>
struct unary_expression<functor_type, transpose> : expression_base<unary_expression<functor_type, transpose>>
{
	functor_type array;

	__BCinline__ static constexpr int DIMS() { return 2; }
	__BCinline__ static constexpr int ITERATOR() { return 2; }
	static_assert(functor_type::DIMS() == 1 || functor_type::DIMS() == 2, "TRANSPOSITION ONLY DEFINED FOR MATRICES AND VECTORS");


	unary_expression(functor_type p) : array(p) {}
	__BCinline__ const auto innerShape() const { return l_array([=](int i) { return i == 0 ? array.cols() : i == 1 ? array.rows() : 1; }); }
	__BCinline__ const auto outerShape() const { return array.outerShape(); }

	__BCinline__ auto operator [](int index) -> decltype(array[index]) {
		return array[DIMS() == 1 ? index : (int)(index / this->rows()) + (index % this->rows()) * this->LD_rows()];
	}
	__BCinline__ auto operator[](int index) const  -> const decltype(array[index])  {
		return array[DIMS() == 1 ? index : (int)(index / this->rows()) + (index % this->rows()) * this->LD_rows()];
	}
	__BCinline__ auto operator ()(int m, int n) const -> decltype(array(m,n)) {
			if (functor_type::ITERATOR() == 0)
				return array(m * array.LD_rows() + n);
			else
				return array(m, n);
	}
	__BCinline__ auto operator ()(int m, int n) -> decltype(array(m,n)) {
			if (functor_type::ITERATOR() == 0)
				return array[m * array.LD_rows() + n];
			else
				return array(m, n);
	}
};
}
#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
