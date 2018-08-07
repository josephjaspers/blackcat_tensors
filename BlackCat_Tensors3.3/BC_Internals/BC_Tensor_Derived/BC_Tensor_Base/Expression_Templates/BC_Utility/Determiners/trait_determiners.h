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
namespace internal {
template<int, class, class> class Array;
}

template<class>class Tensor_Base;

template<class T> struct isPrimaryArray { static constexpr bool conditional = false; };
template<int d, class T, class ml> struct isPrimaryArray<internal::Array<d,T,ml>> { static constexpr bool conditional = true; };
template<class T> static constexpr bool is_array_core() { return isPrimaryArray<T>::conditional; }

template<class> struct determine_functor;
template<class> struct determine_scalar;
template<class> struct determine_iterator;
template<class> struct determine_mathlibrary;
template<class> struct determine_tensor_scalar;

template<class T> using _scalar = typename determine_scalar<std::decay_t<T>>::type;
template<class T> using _mathlib = typename determine_mathlibrary<std::decay_t<T>>::type;
template<class T> using _functor = typename determine_functor<std::decay_t<T>>::type;
template<class T> using _tensor_scalar = typename determine_tensor_scalar<std::decay_t<T>>::type;
//template<class T> static constexpr int _dimension_of  = dimension_of<std::decay_t<T>>;

///DETERMINE_FUNCTOR----------------------------------------------------------------------------------------------
template<class functor>
struct determine_functor<Tensor_Base<functor>>{
	using type = functor;
};

template<int x, class a, class b>
struct determine_functor<internal::Array<x, a, b>>{
	using type = internal::Array<x,a,b>;
};

template<template<class...> class expression, class T, class... set>
struct determine_functor<expression<T, set...>> {
	using type = typename determine_functor<T>::type;
};
//DETERMINE_SCALAR_TYPE----------------------------------------------------------------------------------------------

template<class T>
struct determine_scalar;//  { using type = T; };
template<int dims, class scalar, class ml>
struct determine_scalar<internal::Array<dims, scalar, ml>> {
	using type = scalar;
};

template<template<class...> class expression, class T, class... set>
struct determine_scalar<expression<T, set...>> {
	using type = typename determine_scalar<T>::type;
};

///DETERMINE_MATHLIB---------------------------------------------------------------------------------------
template<class T> struct determine_mathlibrary;
template<class T, class... Ts, template<class...> class list> struct determine_mathlibrary<list<T, Ts...>>
{ using type = typename determine_mathlibrary<T>::type;};

template<int x, class scalar, class ml>
struct determine_mathlibrary<internal::Array<x, scalar, ml>> {
	using type = ml;
};
//template<template<class...> class T, class U, class... set>
//struct determine_mathlibrary<T<U, set...>> {
//	using type = typename determine_mathlibrary<U>::type;
//};

//template<class T, class ml> struct determine_mathlibrary<Scalar<T, ml>> { using type = ml; };
//template<class T, class ml> struct determine_mathlibrary<Vector<T, ml>> { using type = ml; };
//template<class T, class ml> struct determine_mathlibrary<Matrix<T, ml>> { using type = ml; };
//template<class T, class ml> struct determine_mathlibrary<Cube<T, ml>> { using type = ml; };
//template<class T, class ml> struct determine_mathlibrary<Tensor4<T, ml>> { using type = ml; };
//template<class T, class ml> struct determine_mathlibrary<Tensor5<T, ml>> { using type = ml; };

}



#endif /* TRAIT_DETERMINERS_H_ */
