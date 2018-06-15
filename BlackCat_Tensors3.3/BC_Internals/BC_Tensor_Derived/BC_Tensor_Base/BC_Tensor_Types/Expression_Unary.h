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
namespace internal {
template<class value, class operation>
class unary_expression : public expression_base<unary_expression<value, operation>> {
public:

	operation oper;
	value array;

	__BCinline__ static constexpr int DIMS() { return value::DIMS(); }
	__BCinline__ static constexpr int ITERATOR() { return value::ITERATOR(); }
	__BCinline__ static constexpr bool INJECTABLE() { return value::INJECTABLE(); }

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

	//------------------------------------------------------------TREE ROTATION CONSTRUCTORS----------------------------------------------------------------//
	__BC_host_inline__ static constexpr int precedence() { return 1; }/* unary_expressions always have precendence of 1*/
	__BC_host_inline__ static constexpr bool injectable() { return precedence() <= value::precedence() && value::injectable(); }
	__BC_host_inline__ static constexpr bool substituteable() { return false; }	//never fully injectable as we can't "unwrap" the unary _expression

	template<class injection> using type = unary_expression<typename value::template type<injection>, operation>;

	template<class V, class core> //CONVERSION CONSTRUCTOR FOR BLAS ROTATION
	__BCinline__  unary_expression(unary_expression<V, operation> ue, core tensor) : array(ue.array, tensor), oper(ue.oper) {}
	template<class BLAS_expr, int a, int b> //CONVERSION CONSTRUCTOR FOR BLAS ROTATION
	__BCinline__  unary_expression(unary_expression<BLAS_expr, operation> ue, injection_wrapper<value, a, b> tensor) : array(tensor), oper(ue.oper) {
		ue.array.eval(tensor);
	}

	void temporary_destroy() {
		array.temporary_destroy();
	}
};
}
}
#endif /* EXPRESSION_UNARY_POINTWISE_CU_ */
