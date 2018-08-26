/*
 * trait_determiners.h
 *
 *  Created on: May 20, 2018
 *      Author: joseph
 */

#ifndef TRAIT_DETERMINERS_H_
#define TRAIT_DETERMINERS_H_

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
template<int> struct DISABLED;

namespace internal {
template<int, class, class> class Array;
template<int, class, class> class Array_View;

}

template<class>class Tensor_Base;

//template<class T> struct isPrimaryArray { static constexpr bool conditional = false; };
//template<int d, class T, class ml> struct isPrimaryArray<internal::Array<d,T,ml>> { static constexpr bool conditional = true; };
//template<class T> static constexpr bool is_array_core() { return isPrimaryArray<T>::conditional; }

template<class> struct determine_functor;
template<class> struct determine_scalar;
template<class> struct determine_iterator;
template<class> struct determine_mathlibrary;

template<class T> using scalar_of = typename determine_scalar<std::decay_t<T>>::type;
template<class T> using mathlib_of = typename determine_mathlibrary<std::decay_t<T>>::type;
template<class T> using functor_of = typename determine_functor<std::decay_t<T>>::type;

template<class functor>
struct determine_functor<Tensor_Base<functor>>{
	using type = functor;
};

template<int x, class a, class b>
struct determine_functor<internal::Array<x, a, b>>{
	using type = internal::Array<x,a,b>;
};

template<int x, class a, class b>
struct determine_functor<internal::Array_View<x, a, b>>{
	using type = internal::Array_View<x,a,b>;
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

template<int dims, class scalar, class ml>
struct determine_scalar<internal::Array_View<dims, scalar, ml>> {
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

template<int x, class scalar, class ml>
struct determine_mathlibrary<internal::Array_View<x, scalar, ml>> {
	using type = ml;
};

}



#endif /* TRAIT_DETERMINERS_H_ */
