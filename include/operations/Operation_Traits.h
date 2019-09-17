/*
 * Tags.h
 *
 *  Created on: Feb 17, 2019
 *      Author: joseph
 */

#ifndef BC_CORE_OPERATION_TRAITS_TAGS_H_
#define BC_CORE_OPERATION_TRAITS_TAGS_H_

#include "Tags.h"

namespace BC {
namespace oper {
namespace detail {

template<class T> using query_alpha_modifier = typename T::alpha_modifier;
template<class T> using query_beta_modifier = typename T::beta_modifier;
template<class T> using query_dx = decltype(std::declval<T>().dx);

}

template<class T>
struct operation_traits {

	static constexpr int alpha_modifier =
			BC::traits::conditional_detected_t<detail::query_alpha_modifier, T, BC::traits::Integer<1>>::value;

	static constexpr int beta_modifier =
			BC::traits::conditional_detected_t<detail::query_beta_modifier, T, BC::traits::Integer<1>>::value;


	static constexpr bool is_linear_operation            = std::is_base_of<linear_operation,T>::value;
	static constexpr bool is_linear_assignment_operation = std::is_base_of<linear_assignment_operation, T>::value;
	static constexpr bool is_assignment_operation        = std::is_base_of<assignment_operation, T>::value;
	static constexpr bool is_blas_function               = std::is_base_of<BLAS_Function, T>::value;
	static constexpr bool is_nonlinear_operation 		 = !is_blas_function && !is_linear_operation;
};

}
}



#endif /* TAGS_H_ */
