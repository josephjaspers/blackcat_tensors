/*
 * Expression_Binary_Pointwise_ScalarMul.h
 *
 *  Created on: Apr 15, 2018
 *      Author: joseph
 */

#ifndef EXPRESSION_BINARY_POINTWISE_SCALARMUL_H_
#define EXPRESSION_BINARY_POINTWISE_SCALARMUL_H_

#include "BlackCat_Expression_Base.h"
#include "Expression_Binary_Functors.h"

namespace BC {

//class specifically for matrix multiplication
template<class lv, class rv>
struct binary_expression_scalar_mul : Expression_Core_Base<binary_expression_scalar_mul<lv, rv>> {

	mul oper;

	lv left;
	rv right;

	__BCinline__ static constexpr int DIMS() { return lv::DIMS() > rv::DIMS() ? lv::DIMS() : rv::DIMS();}
	__BCinline__ static constexpr int CONTINUOUS() { return max(lv::CONTINUOUS(), rv::CONTINUOUS()); }

	__BCinline__  binary_expression_scalar_mul(lv l, rv r) : left(l), right(r) { }

	__BCinline__  auto operator [](int index) const { return oper(left[index], right[index]); }
	template<class... integers> __BCinline__  auto  operator ()(integers... ints) const { return oper(left(ints...), right(ints...)); }

	__BCinline__ const auto& shape() const { return dominant_type<lv, rv>::shape(left, right); }
	__BCinline__ const auto innerShape() const { return shape().innerShape(); }
	__BCinline__ const auto outerShape() const { return shape().outerShape(); }


	template<class v, class alt> using expr_type = std::conditional_t<v::DIMS() == 0, v, alt>;

	__BCinline__ const auto slice(int i) const {
		return binary_expression_scalar_mul<decltype(left.slice(0)), decltype(right.slice(0))>(left.slice(i), right.slice(i));
	}
	__BCinline__ const auto row(int i) const {
		return binary_expression_scalar_mul<expr_type<lv, decltype(left.row(0))>, expr_type<rv, decltype(right.row(0))>>(left.row(i), right.row(i)); }

	__BCinline__ const auto col(int i) const {
		return binary_expression_scalar_mul<expr_type<lv, decltype(left.col(0))>, expr_type<rv, decltype(right.col(0))>>(left.col(i), right.col(i)); }
};

}

#endif /* EXPRESSION_BINARY_POINTWISE_SCALARMUL_H_ */
