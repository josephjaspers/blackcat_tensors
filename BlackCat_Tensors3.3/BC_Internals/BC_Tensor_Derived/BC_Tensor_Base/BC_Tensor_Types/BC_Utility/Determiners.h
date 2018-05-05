/*
 * Determiners.h
 *
 *  Created on: Feb 25, 2018
 *      Author: joseph
 */

#ifndef DETERMINERS_H_
#define DETERMINERS_H_
#include "MetaTemplateFunctions.h"
#include <type_traits>
namespace BC {

template<bool cond, class a, class b> using ifte = typename std::conditional<cond, a, b>::type;

template<class> class Tensor_Core;
template<class> class Tensor_Slice;
template<class> class Tensor_Scalar;
template<class> class Tensor_Reshape;
template<class,class> class DISABLED;
template<class,class> class Scalar;
template<class,class> class Vector;
template<class,class> class Matrix;
template<class,class> class Cube;
template<class,class> class Tensor4;
template<class,class> class Tensor5;
class BC_Type;

template<class T> struct isCore { static constexpr bool conditional = MTF::isPrimitive<T>::conditional; };
template<template<class> class T, class U> struct isCore<T<U>>
{ static constexpr bool conditional = MTF::isOneOf<T<U>,Tensor_Core<U>, Tensor_Slice<U>, Tensor_Scalar<U>,
	Tensor_Reshape<U>>::conditional; };

template<class T> struct isPrimaryCore { static constexpr bool conditional = MTF::isPrimitive<T>::conditional; };
template<template<class> class T, class U> struct isPrimaryCore<T<U>>
{ static constexpr bool conditional = MTF::same<T<U>,Tensor_Core<U>>::conditional; };

template<class T> static constexpr bool pCore_b = isPrimaryCore<T>::conditional;


template<int> struct base { template<class t, class m> using type = DISABLED<t,m>;	template<class t, class m> using slice = DISABLED<t, m>; };
template<> struct base<0> { template<class t, class m> using type = Scalar<t,m>;  	template<class t, class m> using slice = DISABLED<t, m>; };
template<> struct base<1> { template<class t, class m> using type = Vector<t, m>; 	template<class t,class m> using slice = Scalar<t, m>; };
template<> struct base<2> { template<class t, class m> using type = Matrix<t, m>; 	template<class t,class m> using slice = Vector<t, m>; };
template<> struct base<3> { template<class t, class m> using type = Cube<t, m>;   	template<class t,class m> using slice = Matrix<t, m>; };
template<> struct base<4> { template<class t, class m> using type = Tensor4<t, m>;  template<class t,class m> using slice = Cube<t, m>; };
template<> struct base<5> { template<class t, class m> using type = Tensor5<t, m>;  template<class t,class m> using slice = Tensor4<t, m>; };
template<int x, class a, class b> using rank2class = typename base<x>::template type<a,b>;

template<class> struct ranker;
template<class a, class b> struct ranker<Scalar<a,b>> { static constexpr int value = 0; using ml = b;};
template<class a, class b> struct ranker<Vector<a,b>> { static constexpr int value = 1; using ml = b;};
template<class a, class b> struct ranker<Matrix<a,b>> { static constexpr int value = 2; using ml = b;};
template<class a, class b> struct ranker<Cube<a,b>>   { static constexpr int value = 3; using ml = b;};
template<class a, class b> struct ranker<Tensor4<a,b>>   { static constexpr int value = 4; using ml = b;};
template<class a, class b> struct ranker<Tensor5<a,b>>   { static constexpr int value = 5; using ml = b;};
//
//template<template<class...> class Core, class tensor, class... params> struct ranker<Core<tensor, params...>> {
//	static constexpr int value = ranker<tensor>::value;
//};

template<class T> static constexpr int class2rank = ranker<T>::value;

template<class> struct isTensor {static constexpr bool conditional = false; };
template<class a, class b> struct isTensor<Scalar<a,b>> { static constexpr bool conditional = true; };
template<class a, class b> struct isTensor<Vector<a,b>> { static constexpr bool conditional = true; };
template<class a, class b> struct isTensor<Matrix<a,b>> { static constexpr bool conditional = true; };
template<class a, class b> struct isTensor<Cube<a,b>>   { static constexpr bool conditional = true; };
template<class a, class b> struct isTensor<Tensor4<a,b>>   { static constexpr bool conditional = true; };
template<class a, class b> struct isTensor<Tensor5<a,b>>   { static constexpr bool conditional = true; };
template<class T> static constexpr bool isTensor_b = isTensor<T>::conditional;

template<class> struct determine_functor;
template<class> struct determine_evaluation;
template<class> struct determine_scalar;
template<class> struct determine_iterator;
template<class> struct determine_mathlibrary;
template<class> struct determine_tensor_scalar;


template<class T> using _scalar = typename determine_scalar<T>::type;
template<class T> using _mathlib = typename determine_mathlibrary<T>::type;
template<class T> using _ranker  = typename ranker<T>::type;
template<class T> using _functor = typename determine_functor<T>::type;
template<class T> using _evaluation = typename determine_evaluation<T>::type;
template<class T> static constexpr int _rankOf  = ranker<T>::value;
template<class T> using _iterator = typename determine_iterator<T>::type;
template<class T> using _tensor_scalar = typename determine_tensor_scalar<T>::type;

//remove constness -----------------------------------------------------------------------------------------
namespace rm {
template<class T> struct _const { using type = T; };
template<class T> struct _const<const T> { using type = typename _const<T>::type; };
}
template<class T> using remove_const = typename rm::_const<T>::type;
///DET FUNCTOR----------------------------------------------------------------------------------------------
template<template<class...> class tensor, class functor, class... set>
struct determine_functor<tensor<functor, set...>>{

	using derived = tensor<functor,set...>;
	using type = ifte<std::is_base_of<BC_Type,functor>::value, functor, Tensor_Core<derived>>;
};

///DET EVALUATION----------------------------------------------------------------------------------------------
template<template<class...> class tensor, class functor, class... set>
struct determine_evaluation<tensor<functor, set...>>{
	using derived = tensor<functor,set...>;

	using type = ifte<MTF::isPrimitive<functor>::conditional || isCore<functor>::conditional,
			derived&,
			tensor<_scalar<derived>, _mathlib<derived>>>;
};

//SCALAR_TYPE----------------------------------------------------------------------------------------------
template<class> struct determine_tensor_scalar;


template<class T>
struct determine_scalar {
	static constexpr bool nested_core_type = false;
	using type = T;
};
template<template<class...> class tensor, class T, class... set>
struct determine_scalar<Tensor_Core<tensor<T, set...>>> {
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

///-----

template<template<class...> class tensor, class T, class... set>
struct determine_tensor_scalar<tensor<T, set...>> {
		using type = typename determine_scalar<T>::type;
};

///DETERMINE_ITERATOR---------------------------------------------------------------------------------------
template<class T>
struct determine_iterator {
	using type = decltype(std::declval<T>().getIterator());
};
///DETERMINE_ITERATOR---------------------------------------------------------------------------------------
template<class T, class v=  void>
struct determine_priority {
	static  constexpr int value = 0;
};
template<class T>
struct determine_priority<T, decltype(T::PRIORITY())> {
	static  constexpr int value = T::PRIORITY();
};

template<class T>
static constexpr int _priority = determine_priority<T>::value;
///DETERMINE_MATHLIB
template<class T>
struct determine_mathlibrary {
	using type = std::conditional_t<isTensor_b<T>, MTF::tail<T>, determine_mathlibrary<MTF::head<T>>>;
};

}



#endif /* DETERMINERS_H_ */
