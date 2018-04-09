/*
 * Expression_Unary_Pointwise.cu
 *
 *  Created on: Jan 25, 2018
 *      Author: joseph
 */
#ifndef EXPRESSION_UNARY_POINTWISE_CU_
#define EXPRESSION_UNARY_POINTWISE_CU_

#include "BlackCat_Internal_TypeBase.h"
namespace BC {
template<class T, class operation, class value>
class unary_expression : public Expression_Core_Base<T, unary_expression<T, operation, value>> {
public:

	operation oper;
	value array;

	__BCinline__ static constexpr int DIMS() { return value::DIMS(); }

	__BCinline__  unary_expression(value v, operation op = operation()) : array(v), oper(op) {}

	__BCinline__  const auto innerShape() const 			{ return array.innerShape(); }
	__BCinline__  const auto outerShape() const 			{ return array.outerShape(); }

	__BCinline__ auto operator [](int index) const -> decltype(oper(array[index])) {
		return oper(array[index]);
	}
	__BCinline__ auto operator [](int index) -> decltype(oper(array[index])) {
		return oper(array[index]);
	}

	__BCinline__ const auto slice(int i) const {
		return unary_expression<T, operation, decltype(array.slice(0))>(array.slice(i)); }
	__BCinline__ const auto row(int i) const {
		return unary_expression<T, operation, decltype(array.row(0))>(array.row(i)); }
	__BCinline__ const auto col(int i) const {
		return unary_expression<T, operation, decltype(array.col(0))>(array.col(i)); }
};
}
#endif /* EXPRESSION_UNARY_POINTWISE_CU_ */
