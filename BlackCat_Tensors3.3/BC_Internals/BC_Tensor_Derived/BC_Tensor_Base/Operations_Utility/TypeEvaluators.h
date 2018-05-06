///*
// * TypeEvaluators.h
// *
// *  Created on: May 5, 2018
// *      Author: joseph
// */
//
//#ifndef TYPEEVALUATORS_H_
//#define TYPEEVALUATORS_H_
//
//namespace BC {
//namespace Type_Evaluation {
////determines the return type of pointwise operations
//template<class derived, class param_deriv, class functor, class mathlibrary>
//struct impl {
//	using greater_rank_type = std::conditional_t<(derived::DIMS() > param_deriv::DIMS()), derived, param_deriv>;
//	using param_functor_type = _functor<param_deriv>;
//	using type = 	   expr_sub<greater_rank_type, binary_expression<functor_type ,param_functor_type, functor>, math_library>;
//	using unary_type = expr_sub<greater_rank_type, unary_expression <functor_type, functor>, math_library>;
//};
//
////determines the return type of dot-product operations (and scalar multiplication)
//template<class derived, class param_deriv, class functor, class mathlibrary>
//struct dp_impl {
//	static constexpr bool SCALAR_MUL = derived::DIMS() == 0 || param_deriv::DIMS() == 0;
//	using param_functor_type 	= typename Tensor_Operations<param_deriv>::functor_type;
//	using greater_rank_type 	= std::conditional_t<(derived::DIMS() > param_deriv::DIMS()), derived, param_deriv>;
//	using lesser_rank_type 		= std::conditional_t<(derived::DIMS() < param_deriv::DIMS()), derived, param_deriv>;
//
//	using dot_type 				= binary_expression<_functor<derived>, _functor<param_deriv>, dotproduct<math_library>>;
//	using scalmul_type 			= binary_expression<functor_type , param_functor_type, scalar_mul>;
//
//	using type = std::conditional_t<!SCALAR_MUL,
//					expr_sub<lesser_rank_type, dot_type, math_library>,
//					expr_sub<greater_rank_type, scalmul_type, math_library>>;
//};
//
//}
//}
//
//
//
//
//#endif /* TYPEEVALUATORS_H_ */
