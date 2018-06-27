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

template<class T> struct is_transposed_impl {
	static constexpr bool conditional = false;
};
template<class T, class ml> struct is_transposed_impl<BC::internal::unary_expression<T, oper::transpose<ml>>> {
	static constexpr bool conditional = true;
};
template<class T>
static constexpr bool is_transposed = is_transposed_impl<T>::conditional;

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
		using param_functor_type 	= _functor<param_deriv>;
		using greater_shape 	= std::conditional_t<(derived::DIMS() > param_deriv::DIMS()), derived, param_deriv>;
		using lesser_shape 		= std::conditional_t<(derived::DIMS() < param_deriv::DIMS()), derived, param_deriv>;

		using gemm_t 			= internal::binary_expression<_functor<derived>, _functor<param_deriv>, oper::gemm<mathlib_type>>;
		using gemv_t 			= internal::binary_expression<_functor<derived>, _functor<param_deriv>, oper::gemv<mathlib_type>>;
		using ger_t 			= internal::binary_expression<_functor<derived>, _functor<param_deriv>, oper::ger<mathlib_type>>;

		using axpy_t 			= internal::binary_expression<functor_type , param_functor_type, oper::scalar_mul>;

		static constexpr bool axpy = derived::DIMS() == 0 || param_deriv::DIMS() == 0;
		static constexpr bool gemm = (derived::DIMS() == 2 && param_deriv::DIMS() == 2);
		static constexpr bool gemv = derived::DIMS() == 2 && param_deriv::DIMS() == 1;
		static constexpr bool ger  = derived::DIMS() == 1 && param_deriv::DIMS() == 1;

		using type = std::conditional_t<axpy, expressionSubstitution<lesser_shape, axpy_t, mathlib_type>,
					std::conditional_t<gemm, expressionSubstitution<greater_shape, gemm_t, mathlib_type>,
					std::conditional_t<gemv, expressionSubstitution<lesser_shape,  gemv_t, mathlib_type>,
					std::conditional_t<ger, tensor_of_t<2, ger_t, mathlib_type>, void>>>>;
	};

};
}
}

#endif /* TYPE_EVALUATOR_H_ */
