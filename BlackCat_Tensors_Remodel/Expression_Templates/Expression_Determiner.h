/*
 * Type_Evaluator.h
 *
 *  Created on: May 6, 2018
 *      Author: joseph
 */

#ifndef TYPE_EVALUATOR_H_
#define TYPE_EVALUATOR_H_

namespace BC {

template<class> class Tensor_Base;

template<class derived>
struct expression_determiner {

	using functor_type 		= functor_of<derived>;
	using scalar_type 		= scalar_of<derived>;
	using mathlib_type 		= mathlib_of<derived>;

	//determines the return type of pointwise operations
	template<class param_deriv, class functor>
	struct impl {
		using greater_rank_type = std::conditional_t<(derived::DIMS() > param_deriv::DIMS()), derived, param_deriv>;
		using paramfunctor_of_type = functor_of<param_deriv>;
		using type = 	   Tensor_Base<internal::binary_expression<functor_type ,paramfunctor_of_type, functor>>;
		using unary_type = Tensor_Base<internal::unary_expression <functor_type, functor>>;
	};

	//determines the return type of dot-product operations (and scalar multiplication)
	template<class param_deriv>
	struct dp_impl {
		using paramfunctor_of_type 	= functor_of<param_deriv>;
		using greater_shape 	= std::conditional_t<(derived::DIMS() > param_deriv::DIMS()), derived, param_deriv>;
		using lesser_shape 		= std::conditional_t<(derived::DIMS() < param_deriv::DIMS()), derived, param_deriv>;

		using gemm_t 			= internal::binary_expression<functor_of<derived>, functor_of<param_deriv>, oper::gemm<mathlib_type>>;
		using gemv_t 			= internal::binary_expression<functor_of<derived>, functor_of<param_deriv>, oper::gemv<mathlib_type>>;
		using ger_t 			= internal::binary_expression<functor_of<derived>, functor_of<param_deriv>, oper::ger<mathlib_type>>;

		using axpy_t 			= internal::binary_expression<functor_type , paramfunctor_of_type, oper::scalar_mul>;

		static constexpr bool axpy = derived::DIMS() == 0 || param_deriv::DIMS() == 0;
		static constexpr bool gemm = (derived::DIMS() == 2 && param_deriv::DIMS() == 2);
		static constexpr bool gemv = derived::DIMS() == 2 && param_deriv::DIMS() == 1;
		static constexpr bool ger  = derived::DIMS() == 1 && param_deriv::DIMS() == 1;

		using type = std::conditional_t<axpy, Tensor_Base<axpy_t>,
					std::conditional_t<gemm, Tensor_Base<gemm_t>,
					std::conditional_t<gemv, Tensor_Base<gemv_t>,
					std::conditional_t<ger, Tensor_Base<ger_t>, void>>>>;
	};

};
}

#endif /* TYPE_EVALUATOR_H_ */
