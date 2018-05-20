/*
 * trait_determiners.h
 *
 *  Created on: May 20, 2018
 *      Author: joseph
 */

#ifndef TRAIT_DETERMINERS_H_
#define TRAIT_DETERMINERS_H_

#include "type_determiners.h"

/*
 * Determines the primary traits of an internal-type
 * functor
 * scalar
 * iterator
 * mathlibrary
 * --tensor_scalar is the specific function determining the scalar of a tensor (opposed to internal-tensor type)
 * --IE tensor_scalar<Matrix<A,B>>, while _scalar is for... _scalar<binary_expression<lv,rv,oper>>
 */

namespace BC {
template<class> struct determine_functor;
template<class> struct determine_scalar;
template<class> struct determine_iterator;
template<class> struct determine_mathlibrary;
template<class> struct determine_tensor_scalar;

template<class T> using _scalar = typename determine_scalar<T>::type;
template<class T> using _mathlib = typename determine_mathlibrary<T>::type;
template<class T> using _functor = typename determine_functor<T>::type;
template<class T> using _iterator = typename determine_iterator<T>::type;
template<class T> using _tensor_scalar = typename determine_tensor_scalar<T>::type;
template<class T> static constexpr int _dimension_of  = dimension_of<T>::value;

///DETERMINE_FUNCTOR----------------------------------------------------------------------------------------------
template<template<class...> class tensor, class functor, class... set>
struct determine_functor<tensor<functor, set...>>{

	using derived = tensor<functor,set...>;
	using type = std::conditional_t<std::is_base_of<BC_Type,functor>::value, functor, Core<derived>>;
};

//DETERMINE_SCALAR_TYPE----------------------------------------------------------------------------------------------
template<class> struct determine_tensor_scalar;

template<class T>
struct determine_scalar {
	static constexpr bool nested_core_type = false;
	using type = T;
};
template<template<class...> class tensor, class T, class... set>
struct determine_scalar<Core<tensor<T, set...>>> {
	static constexpr bool nested_core_type = true;
	using type = typename determine_scalar<T>::type;
};

template<template<class...> class expression, class T, class... set>
struct determine_scalar<expression<T, set...>> {
	static constexpr bool nested_core_type = determine_scalar<T>::nested_core_type;
	using type = std::conditional_t<nested_core_type, typename determine_scalar<T>::type,

			std::conditional_t<isTensor_b<expression<T, set...>>, T,

			expression<T, set...>>>;
};
///DETERMINE_ITERATOR---------------------------------------------------------------------------------------
template<class T>
struct determine_iterator {
	using type = decltype(std::declval<T>().getIterator());
};

///DETERMINE_MATHLIB---------------------------------------------------------------------------------------
template<class T>
struct determine_mathlibrary {
	using type = std::conditional_t<isTensor_b<T>, MTF::tail<T>, determine_mathlibrary<MTF::head<T>>>;
};






}



#endif /* TRAIT_DETERMINERS_H_ */
