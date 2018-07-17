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
namespace oper {
template<class ml> class transpose;
}


namespace internal {

template<class functor_type, class ml>
struct unary_expression<functor_type, oper::transpose<ml>> : expression_base<unary_expression<functor_type, oper::transpose<ml>>>
{
	functor_type array;

	__BCinline__ static constexpr int DIMS() { return functor_type::DIMS(); }
	__BCinline__ static constexpr int ITERATOR() { return DIMS(); }

	unary_expression(functor_type p) : array(p) {}

	__BCinline__ const auto inner_shape() const {
		return l_array<DIMS()>([=](int i) {
			if (DIMS() >= 2)
				return i == 0 ? array.cols() : i == 1 ? array.rows() : array.dimension(i);
			else if (DIMS() == 2)
				return i == 0 ? array.cols() : i == 1 ? array.rows() : 1;
			else if (DIMS() == 1)
				return i == 0 ? array.rows() : 1;
			else
				return 1;
		});
	}
	__BCinline__ int size() const { return array.size(); }
	__BCinline__ int rows() const { return array.cols(); }
	__BCinline__ int cols() const { return array.rows(); }
	__BCinline__ int dimension(int i) const { return i == 0 ? array.cols() : i == 1 ? array.rows() : array.dimension(i); }
	__BCinline__ const auto outer_shape() const { return array.outer_shape(); }

	__BCinline__ auto operator [](int index) -> decltype(array[index]) {
		return array[DIMS() == 1 ? index : (int)(index / this->rows()) + (index % this->rows()) * this->leading_dimension(0)];
	}
	__BCinline__ auto operator[](int index) const  -> const decltype(array[index])  {
		return array[DIMS() == 1 ? index : (int)(index / this->rows()) + (index % this->rows()) * this->leading_dimension(0)];
	}
	template<class... ints>
	__BCinline__ auto operator ()(int m, int n, ints... integers) const -> decltype(array(n,m)) {
		return array(n,m, integers...);
	}
	template<class... ints>
	__BCinline__ auto operator ()(int m, int n, ints... integers) -> decltype(array(n,m)) {
		return array(n, m, integers...);
	}
};
}
}
#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
