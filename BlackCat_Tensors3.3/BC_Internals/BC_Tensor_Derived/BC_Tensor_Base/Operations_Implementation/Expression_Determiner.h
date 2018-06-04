/*
 * Type_Evaluator.h
 *
 *  Created on: May 6, 2018
 *      Author: joseph
 */

#ifndef TYPE_EVALUATOR_H_
#define TYPE_EVALUATOR_H_

namespace BC {
namespace operationImpl {
template<class derived>
struct expression_determiner {

	using functor_type 		= _functor<derived>;
	using scalar_type 		= _scalar<derived>;
	using mathlib_type 		= _mathlib<derived>;

	template<class shell, class... T> using expressionSubstitution = typename MTF::shell_of<shell>::template type<T...>;

	//determines the return type of pointwise operations
	template<class param_deriv, class functor>
	struct impl {
		using greater_rank_type = std::conditional_t<(derived::DIMS() > param_deriv::DIMS()), derived, param_deriv>;
		using param_functor_type = _functor<param_deriv>;
		using type = 	   expressionSubstitution<greater_rank_type, internal::binary_expression<functor_type ,param_functor_type, functor>, mathlib_type>;
		using unary_type = expressionSubstitution<greater_rank_type, internal::unary_expression <functor_type, functor>, mathlib_type>;
	};

	//determines the return type of dot-product operations (and scalar multiplication)
	template<class param_deriv>
	struct dp_impl {
		static constexpr bool SCALAR_MUL = derived::DIMS() == 0 || param_deriv::DIMS() == 0;
		using param_functor_type 	= _functor<param_deriv>;
		using greater_shape 	= std::conditional_t<(derived::DIMS() > param_deriv::DIMS()), derived, param_deriv>;
		using lesser_shape 		= std::conditional_t<(derived::DIMS() < param_deriv::DIMS()), derived, param_deriv>;

		using dot_type 				= internal::binary_expression<_functor<derived>, _functor<param_deriv>, function::dotproduct<mathlib_type>>;
		using scalmul_type 			= internal::binary_expression<functor_type , param_functor_type, function::scalar_mul>;

		using type = std::conditional_t<!SCALAR_MUL,
						expressionSubstitution<lesser_shape, dot_type, mathlib_type>,
						expressionSubstitution<greater_shape, scalmul_type, mathlib_type>>;
	};

};
}
}

#endif /* TYPE_EVALUATOR_H_ */
