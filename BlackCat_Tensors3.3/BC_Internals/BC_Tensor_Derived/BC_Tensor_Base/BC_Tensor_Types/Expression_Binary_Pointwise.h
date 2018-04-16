/*
 * BC_Expression_Binary.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */
#ifndef EXPRESSION_BINARY_POINTWISE_SAME_H_
#define EXPRESSION_BINARY_POINTWISE_SAME_H_

#include "BlackCat_Internal_Type_ExpressionBase.h"

namespace BC {

template<class lv, class rv, class operation>
struct binary_expression : public Expression_Core_Base<binary_expression<lv, rv, operation>> {

	operation oper;

	lv left;
	rv right;

	__BCinline__ static constexpr int DIMS() { return lv::DIMS() > rv::DIMS() ? lv::DIMS() : rv::DIMS();}
	__BCinline__  binary_expression(lv l, rv r, operation oper_ = operation()) : left(l), right(r), oper(oper_) {}

	__BCinline__  auto  operator [](int index) const { return oper(left[index], right[index]); }

	__BCinline__ const auto& shape() const { return dominant_type<lv, rv>::shape(left, right); }
	__BCinline__ const auto  innerShape() const { return shape().innerShape(); }
	__BCinline__ const auto  outerShape() const { return shape().outerShape(); }


	template<class v, class alt>
	using expr_type = std::conditional_t<v::DIMS() == 0, v, alt>;

	__BCinline__ const auto slice(int i) const {
		return binary_expression<decltype(left.slice(0)), decltype(right.slice(0)), operation>(left.slice(i), right.slice(i));
	}
	__BCinline__ const auto row(int i) const {
		return binary_expression<expr_type<lv, decltype(left.row(0))>, expr_type<rv, decltype(right.row(0))>, operation>(left.row(i), right.row(i)); }

	__BCinline__ const auto col(int i) const {
		return binary_expression<expr_type<lv, decltype(left.col(0))>, expr_type<rv, decltype(right.col(0))>, operation>(left.col(i), right.col(i)); }
};

}

#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */

