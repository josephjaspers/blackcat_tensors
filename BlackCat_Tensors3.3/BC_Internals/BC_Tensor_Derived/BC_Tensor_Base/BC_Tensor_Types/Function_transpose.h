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

	__BCinline__ static constexpr int DIMS() { return 2; }
	__BCinline__ static constexpr int ITERATOR() { return 2; }
	static_assert(functor_type::DIMS() == 1 || functor_type::DIMS() == 2, "TRANSPOSITION ONLY DEFINED FOR MATRICES AND VECTORS");


	unary_expression(functor_type p) : array(p) {}

	//blas injection
	template<class core> unary_expression(functor_type ary, core tensor) : array(ary, tensor) {}
	template<class BLAS_expr> //CONVERSION CONSTRUCTOR FOR BLAS ROTATION
		__BCinline__  unary_expression(unary_expression<BLAS_expr, oper::transpose<ml>> ue, functor_type tensor) : array(tensor) {
		ue.array.eval(tensor);
	}

	__BCinline__ const auto inner_shape() const { return l_array([=](int i) { return i == 0 ? array.cols() : i == 1 ? array.rows() : 1; }); }
	__BCinline__ const auto outer_shape() const { return array.outer_shape(); }

	__BCinline__ auto operator [](int index) -> decltype(array[index]) {
		return array[DIMS() == 1 ? index : (int)(index / this->rows()) + (index % this->rows()) * this->ld1()];
	}
	__BCinline__ auto operator[](int index) const  -> const decltype(array[index])  {
		return array[DIMS() == 1 ? index : (int)(index / this->rows()) + (index % this->rows()) * this->ld1()];
	}
	__BCinline__ auto operator ()(int m, int n) const -> decltype(array(m,n)) {
			if (functor_type::ITERATOR() == 0)
				return array(m * array.ld1() + n);
			else
				return array(m, n);
	}
	__BCinline__ auto operator ()(int m, int n) -> decltype(array(m,n)) {
			if (functor_type::ITERATOR() == 0)
				return array[m * array.ld1() + n];
			else
				return array(m, n);
	}


	//change this to true once we add the BLAS transpose (cublas has built in transpose, need to write CPU)
	static constexpr bool injectable() { return functor_type::injectable(); }
	static constexpr int precedence() { return -1; }
	static constexpr bool substituteable() { return false; }

	template<class inj_t, bool presub> using sub_t = std::conditional_t<presub, void, inj_t>;
	template<class injection, bool presub = false> using type = unary_expression<typename functor_type::template type<sub_t< injection, presub>>, oper::transpose<ml>>;

	void temporary_destroy() {
		array.temporary_destroy();
	}
};
}
}
#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
