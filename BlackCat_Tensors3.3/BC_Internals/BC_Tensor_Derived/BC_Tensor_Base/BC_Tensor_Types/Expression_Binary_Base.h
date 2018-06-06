/*
 * BC_Expression_Binary.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */
#ifndef EXPRESSION_BINARY_POINTWISE_SAME_H_
#define EXPRESSION_BINARY_POINTWISE_SAME_H_

#include "Expression_Base.h"

namespace BC {
namespace internal {
template<class lv, class rv, class operation>
struct binary_expression : public expression_base<binary_expression<lv, rv, operation>> {

	operation oper;

	lv left;
	rv right;

	__BCinline__ static constexpr int DIMS() { return lv::DIMS() > rv::DIMS() ? lv::DIMS() : rv::DIMS();}
	__BCinline__ static constexpr int ITERATOR() { return max(lv::ITERATOR(), rv::ITERATOR()); }
	__BCinline__ static constexpr bool INJECTABLE() { return lv::INJECTABLE() || rv::INJECTABLE(); }

	__BCinline__  binary_expression(lv l, rv r, operation oper_ = operation()) : left(l), right(r), oper(oper_) {}

//		template<class R, class core> 	__BC_host_inline__//function replacement right
//	 	binary_expression(binary_expression<lv, R, operation> expr, core tensor_core) : left(expr.left), right(expr.right, tensor_core) {}
//	 	template<class L, class core> 	__BC_host_inline__//function replacement left
//	 	binary_expression(binary_expression<L, rv, operation> expr, core tensor_core) : left(expr.left, tensor_core), right(expr.right){}
//
//
//	 	template<class BLAS_expression> __BC_host_inline__//function injection right
//	 	 binary_expression(binary_expression<lv, BLAS_expression, operation> expr, rv tensor) : left(expr.left), right(tensor) {
//	 		expr.right.eval(tensor); //extract right hand side, which represents a BLAS function, inject the tensor into the output
//	 	}
//	 	template<class BLAS_expression> __BC_host_inline__//function injection left
//	 	binary_expression(binary_expression<BLAS_expression, rv, operation> expr, lv tensor) : left(tensor), right(expr.right) {
//	 		expr.left.eval(tensor); //extract right hand side, which represents a BLAS function, inject the tensor into the output
//	 	}

	__BCinline__  auto  operator [](int index) const { return oper(left[index], right[index]); }
	template<class... integers> __BCinline__  auto  operator ()(integers... ints) const { return oper(left(ints...), right(ints...)); }

	__BCinline__ const auto& shape() const { return dominant_type<lv, rv>::shape(left, right); }
	__BCinline__ const auto  inner_shape() const { return shape().inner_shape(); }
	__BCinline__ const auto  outer_shape() const { return shape().outer_shape(); }


	__BCinline__ const auto slice(int i) const {
		return binary_expression<decltype(left.slice(0)), decltype(right.slice(0)), operation>(left.slice(i), right.slice(i));}
	__BCinline__ const auto row(int i) const {
		return binary_expression<decltype(left.row(0)), decltype(right.row(0)), operation>(left.row(i), right.row(i)); }
	__BCinline__ const auto col(int i) const {
		return binary_expression<decltype(left.col(0)), decltype(right.col(0)), operation>(left.col(i), right.col(i)); }
	__BCinline__ const auto scalar(int i) const {
		return binary_expression<decltype(left.scalar(0)), decltype(right.scalar(0)), operation>(left.col(i), right.col(i)); }

	void eval() const {
		left.eval();
		right.eval();
	}
};
}
}

#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */

