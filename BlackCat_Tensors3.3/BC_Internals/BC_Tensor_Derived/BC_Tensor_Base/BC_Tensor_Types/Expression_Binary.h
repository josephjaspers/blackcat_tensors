/*
 * BC_Expression_Binary.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */
#ifndef EXPRESSION_BINARY_POINTWISE_SAME_H_
#define EXPRESSION_BINARY_POINTWISE_SAME_H_

#include "Expression_Base.h"
#include "Parse_Tree_Functions.h"


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

	__BCinline__  auto  operator [](int index) const { return oper(left[index], right[index]); }
	template<class... integers> __BCinline__  auto  operator ()(integers... ints) const { return oper(left(ints...), right(ints...)); }

	__BCinline__ const auto& shape() const { return dominant_type<lv, rv>::shape(left, right); }
	__BCinline__ const auto  inner_shape() const { return shape().inner_shape(); }
	__BCinline__ const auto  outer_shape() const { return shape().outer_shape(); }


	//------------------------------------------------------------TREE ROTATION CONSTRUCTORS----------------------------------------------------------------//
	__BC_host_inline__ static constexpr int precedence() { return tree::precedence<operation>(); }
	__BC_host_inline__ static constexpr bool lv_injectable() { return precedence() <= lv::precedence() && lv::injectable(); }
	__BC_host_inline__ static constexpr bool rv_injectable() { return precedence() <= rv::precedence() && rv::injectable(); }
	__BC_host_inline__ static constexpr bool lv_substituteable() { return precedence() <= lv::precedence() && lv::injectable(); }
	__BC_host_inline__ static constexpr bool rv_substituteable() { return precedence() <= rv::precedence() && rv::injectable(); }
	__BC_host_inline__ static constexpr bool injectable()	  { return lv_injectable() ||  rv_injectable(); }
	__BC_host_inline__ static constexpr bool substituteable() { return lv::substituteable() && rv::substituteable(); }

	__BC_host_inline__ static constexpr int alpha_mod() { return tree::alpha_of<operation>(); }
	__BC_host_inline__ static constexpr int beta_mod() { return tree::beta_of<operation>(); }

	template<class injection> using lv_inj_t = typename lv::template type<injection>;
	template<class injection> using rv_inj_t = typename rv::template type<injection>;

	using default_type = std::conditional_t<tree::injectable_assignment<operation>(), lv, void>;

	template<class injection = default_type>
	using type =
			std::conditional_t<substituteable(),
			injection,
			binary_expression<lv_inj_t<injection>, rv_inj_t<injection>, operation>>;

	using injection_type = type<default_type>;

	//Right side replacement
	template<class R, class core, int a, int b> 	__BC_host_inline__
	binary_expression(binary_expression<lv, R, operation> expr, injection_wrapper<core, a, b> tensor_core) : left(expr.left), right(expr.right, tensor_core) {}

	//Left side replacement
	template<class L, class core, int a, int b> 	__BC_host_inline__
	binary_expression(binary_expression<L, rv, operation> expr, injection_wrapper<core, a, b> tensor_core)
	: left(expr.left, tensor_core), right(expr.right){}


	//right side injection
	template<class expr_t, int a, int b> __BC_host_inline__
	 binary_expression(binary_expression<lv, expr_t, operation> expr, injection_wrapper<rv, a, b> tensor) : left(expr.left), right((rv&)tensor) {
		expr.right.eval(tensor); //extract right hand side, which represents a BLAS function, inject the tensor into the output
	}

	//left side injection
	template<class expr_t, int a , int b> __BC_host_inline__ //function injection left
	binary_expression(binary_expression<expr_t, rv, operation> expr, injection_wrapper<lv, a, b> tensor) : left((lv&)tensor), right(expr.right) {
		expr.left.eval(tensor); //extract right hand side, which represents a BLAS function, inject the tensor into the output
	}

	template<class core, int a, int b>
	std::enable_if<std::is_base_of<BLAS_FUNCTION, lv>::value && std::is_base_of<BLAS_FUNCTION, rv>::value>
	eval(injection_wrapper<core, a, b> injection) {
		left.eval(injection);
		right.eval(injection_wrapper<core, tree::alpha_of<operation>(), tree::beta_of<operation>()>(injection.data())); //we wrap data to ensure scalar's are not calculated twice
	}

	__BC_host_inline__ void temporary_destroy() {
		left.temporary_destroy();
		right.temporary_destroy();
	}
};
}
}


#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */

