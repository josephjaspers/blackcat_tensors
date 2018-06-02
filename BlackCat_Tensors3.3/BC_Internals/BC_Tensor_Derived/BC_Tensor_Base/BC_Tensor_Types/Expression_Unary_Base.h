/*
 * Expression_Unary_Base.cu
 *
 *  Created on: Jan 25, 2018
 *      Author: joseph
 */
#ifndef EXPRESSION_UNARY_POINTWISE_CU_
#define EXPRESSION_UNARY_POINTWISE_CU_

#include "Expression_Base.h"

namespace BC {
template<class value, class operation>
class unary_expression : public expression_base<unary_expression<value, operation>> {
public:

	operation oper;
	value array;

	__BCinline__ static constexpr int DIMS() { return value::DIMS(); }
	__BCinline__ static constexpr int ITERATOR() { return value::ITERATOR(); }

	__BCinline__  unary_expression(value v, operation op = operation()) : array(v), oper(op) {}

	__BCinline__  const auto inner_shape() const 			{ return array.inner_shape(); }
	__BCinline__  const auto outer_shape() const 			{ return array.outer_shape(); }

	__BCinline__ auto operator [](int index) const -> decltype(oper(array[index])) {
		return oper(array[index]);
	}
	__BCinline__ auto operator [](int index) -> decltype(oper(array[index])) {
		return oper(array[index]);
	}
	template<class... integers>__BCinline__ auto operator ()(integers... index) const -> decltype(oper(array(index...))) {
		return oper(array(index...));
	}
	template<class... integers>	__BCinline__ auto operator ()(integers... index) -> decltype(oper(array(index...))) {
		return oper(array(index...));
	}
	__BCinline__ const auto slice(int i) const {
		return unary_expression<decltype(array.slice(0)), operation>(array.slice(i)); }
	__BCinline__ const auto row(int i) const {
		return unary_expression<decltype(array.row(0)),operation>(array.row(i)); }
	__BCinline__ const auto col(int i) const {
		return unary_expression<decltype(array.col(0)),operation>(array.col(i)); }
	__BCinline__ const auto scalar(int i) const {
		return unary_expression<decltype(array.scalar(0)),operation>(array.scalar(i)); }
};
}
#endif /* EXPRESSION_UNARY_POINTWISE_CU_ */
