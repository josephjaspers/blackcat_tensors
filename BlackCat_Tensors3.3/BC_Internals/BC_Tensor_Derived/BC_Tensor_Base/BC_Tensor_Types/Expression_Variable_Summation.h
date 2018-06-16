///*
// * Expression_Variable_Summation.h
// *
// *  Created on: Jun 15, 2018
// *      Author: joseph
// */
//
//#ifndef EXPRESSION_VARIABLE_SUMMATION_H_
//#define EXPRESSION_VARIABLE_SUMMATION_H_
//
//namespace BC{
//namespace internal {
//
//template<class oper> static constexpr bool is_blas = std::is_base_of<BLAS_FUNCTION, oper>::value;
//
//template<class.... Ts>
//struct gather_non_blas {
//
//};
//
//template<class... exprs>
//struct variable_expression : public expression_base<variable_expression<exprs...>> {
//
//	__BCinline__ static constexpr int DIMS() { return -1; }
//	__BCinline__ static constexpr int ITERATOR() { return -1; }
//	__BCinline__ static constexpr bool INJECTABLE() { return true; }
//
//};
//
//template<class expr, class... exprs>
//struct variable_expression<expr, exprs...>
//	: public expression_base<variable_expression<expr, exprs...>>,
//	  public variable_expression<exprs...> {
//
//	using next = variable_expression<exprs...>;
//
//	expr ex;
//
//	variable_expression(expr x_, exprs... xs) : next(xs...) {}
//
//	__BCinline__ static constexpr int DIMS() { return max(expr::DIMS(), next::DIMS()); }
//	__BCinline__ static constexpr int ITERATOR() { return max(expr::ITERATOR(), next::ITERATOR()); }
//	__BCinline__ static constexpr bool INJECTABLE() { return true; }
//
//
//};
//
//
//}
//}
//
//
//#endif /* EXPRESSION_VARIABLE_SUMMATION_H_ */
