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
namespace {

using BC::meta::conditional_detected_t;
using BC::meta::is_detected_v;
using BC::meta::Integer;
using std::is_base_of;

template<class T> using query_alpha_modifier = typename T::alpha_modifier;
template<class T> using query_beta_modifier = typename T::beta_modifier;

template<class T> using query_dx = decltype(std::declval<T>().dx);
template<class T> using query_cached_dx = decltype(std::declval<T>().cached_dx);

template<class T> using query_dx = decltype(std::declval<T>().dx);

}


template<class T>
struct operation_traits {

	 static constexpr bool dx_is_defined =  BC::meta::is_detected_v<query_dx, T>;
	 static auto select_on_dx(T&& expression) {
		 return BC::meta::constexpr_ternary<dx_is_defined>(
		 		BC::meta::bind([&](auto&& expression) {
			 	 	 return expression.dx;
		 	 	 }, expression),
		 		[&]() { return expression; }
		 	 );
	 }

	static constexpr int alpha_modifier =
			conditional_detected_t<query_alpha_modifier, T, Integer<1>>::value;
	static constexpr int beta_modifier =
				conditional_detected_t<query_beta_modifier, T, Integer<1>>::value;

	static constexpr bool has_dx = is_detected_v<query_dx, T>;
	static constexpr bool has_cached_dx = is_detected_v<query_cached_dx, T>;

	static constexpr bool is_linear_operation            = is_base_of<linear_operation,T>::value;
	static constexpr bool is_linear_assignment_operation = is_base_of<linear_assignment_operation, T>::value;
	static constexpr bool is_assignment_operation        = is_base_of<assignment_operation, T>::value;
	static constexpr bool is_blas_function               = is_base_of<BLAS_Function, T>::value;
	static constexpr bool is_nonlinear_operation 		 = !is_blas_function && !is_linear_operation;
};

}
}



#endif /* TAGS_H_ */
