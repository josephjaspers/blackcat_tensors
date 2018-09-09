/*
 * Tensor_Common.h
 *
 *  Created on: Sep 9, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_
#define BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_

#define BC_ARRAY_ONLY(literal) static_assert(true, " ");//static_assert(BC::is_array<functor_of<derived>>(), "BC Method: '" literal "' is only supported by Array_Base classes")

#include <type_traits>
namespace BC {
template<int> class DISABLED;

}
//namespace BC {
//template<int> struct DISABLED;
//
//namespace internal {
//template<int, class, class> class Array;
//template<int, class, class> class Array_View;
//
//}
//
//template<class>class Tensor_Base;
//template<class> struct determine_functor;
//template<class> struct determine_scalar;
//template<class> struct determine_iterator;
//template<class> struct determine_mathlibrary;
//
//template<class T> using scalar_of = typename determine_scalar<std::decay_t<T>>::type;
//template<class T> using mathlib_of = typename determine_mathlibrary<std::decay_t<T>>::type;
//template<class T> using functor_of = typename determine_functor<std::decay_t<T>>::type;
//
//template<class functor>
//struct determine_functor<Tensor_Base<functor>>{
//	using type = functor;
//};
//
//template<int x, class a, class b>
//struct determine_functor<internal::Array<x, a, b>>{
//	using type = internal::Array<x,a,b>;
//};
//
//template<int x, class a, class b>
//struct determine_functor<internal::Array_View<x, a, b>>{
//	using type = internal::Array_View<x,a,b>;
//};
//
//template<template<class...> class expression, class T, class... set>
//struct determine_functor<expression<T, set...>> {
//	using type = typename determine_functor<T>::type;
//};
////DETERMINE_SCALAR_TYPE----------------------------------------------------------------------------------------------
//
//template<class T>
//struct determine_scalar;//  { using type = T; };
//template<int dims, class scalar, class ml>
//struct determine_scalar<internal::Array<dims, scalar, ml>> {
//	using type = scalar;
//};
//
//template<int dims, class scalar, class ml>
//struct determine_scalar<internal::Array_View<dims, scalar, ml>> {
//	using type = scalar;
//};
//
//
//template<template<class...> class expression, class T, class... set>
//struct determine_scalar<expression<T, set...>> {
//	using type = typename determine_scalar<T>::type;
//};
//
/////DETERMINE_MATHLIB---------------------------------------------------------------------------------------
//template<class T> struct determine_mathlibrary;
//template<class T, class... Ts, template<class...> class list> struct determine_mathlibrary<list<T, Ts...>>
//{ using type = typename determine_mathlibrary<T>::type;};
//
//template<int x, class scalar, class ml>
//struct determine_mathlibrary<internal::Array<x, scalar, ml>> {
//	using type = ml;
//};
//
//template<int x, class scalar, class ml>
//struct determine_mathlibrary<internal::Array_View<x, scalar, ml>> {
//	using type = ml;
//};

//}


#endif /* BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_ */
