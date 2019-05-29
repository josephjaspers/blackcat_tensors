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

template<class T, class voider=void>
struct detect_forward_propagate {
	static T&& apply(T&& expression) {
		return expression;
	}
};
template<class T>
struct detect_forward_propagate<T, std::enable_if_t<BC::meta::true_v<decltype(std::declval<T>().forward_propagate())>>> {
	static T&& apply(T&& expression){
		return expression.forward_propagate();
	}
};
template<class T, class voider=void>
struct detect_backward_propagate {
	static T&& apply(T&& expression) {
		return expression;
	}
};
template<class T>
struct detect_backward_propagate<T, std::enable_if_t<BC::meta::true_v<decltype(std::declval<T>().backward_propagate())>>> {
	static T&& apply(T&& expression){
		return expression.backward_propagate();
	}
};

}


template<class T>
struct operation_traits {
	static constexpr int alpha_modifier = std::is_base_of<alpha_modifier_base, T>::value ? T::alpha_mod : 1;
	static constexpr int beta_modifier = std::is_base_of<beta_modifier_base, T>::value ? T::beta_mod : 1;
	static constexpr bool is_linear_operation = std::is_base_of<linear_operation,T>::value;
	static constexpr bool is_linear_assignment_operation = std::is_base_of<linear_assignment_operation, T>::value;
	static constexpr bool is_assignment_operation        = std::is_base_of<assignment_operation, T>::value;
	static constexpr bool is_blas_function               = std::is_base_of<BLAS_Function, T>::value;
	static constexpr bool is_nonlinear_operation 		 = !is_blas_function && !is_linear_operation;

	auto select_on_forward_propagate(T&& expression) {
		return detail::detect_forward_propagate<T>::apply(expression);
	}

	auto select_on_backward_propagate(T&& expression) {
		return detail::detect_backward_propagate<T>::apply(expression);
	}

};


}
}



#endif /* TAGS_H_ */
