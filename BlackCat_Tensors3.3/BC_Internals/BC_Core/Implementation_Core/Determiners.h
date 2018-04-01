/*
 * Determiners.h
 *
 *  Created on: Feb 25, 2018
 *      Author: joseph
 */

#ifndef DETERMINERS_H_
#define DETERMINERS_H_
#include <type_traits>
namespace BC {

template<bool cond, class a, class b> using ifte = typename std::conditional<cond, a, b>::type;

template<class> class Tensor_Core;
template<class> class Tensor_Slice;
template<class> class Tensor_Scalar;
template<class, int> class Tensor_Reshape;

template<class,class> class DISABLED;
template<class,class> class Scalar;
template<class,class> class Vector;
template<class,class> class Matrix;
template<class,class> class Cube;
template<class,class> class Tensor4;
template<class,class> class Tensor5;

template<class T> struct isCore { static constexpr bool conditional = MTF::isPrimitive<T>::conditional; };
template<template<class> class T, class U> struct isCore<T<U>>
{ static constexpr bool conditional = MTF::isOneOf<T<U>,Tensor_Core<U>, Tensor_Slice<U>, Tensor_Scalar<U>,
	Tensor_Reshape<U,0>,Tensor_Reshape<U,1>,Tensor_Reshape<U,2>,Tensor_Reshape<U,3>,Tensor_Reshape<U,4>,Tensor_Reshape<U,5>>::conditional; };




template<class> struct ranker;
template<int> struct base;
template<int x> struct base { template<class t, class m> using type = DISABLED<t,m>;  template<class t, class m> using slice = DISABLED<t, m>; };
template<> struct base<0> { template<class t, class m> using type = Scalar<t,m>;  template<class t, class m> using slice = DISABLED<t, m>; };
template<> struct base<1> { template<class t, class m> using type = Vector<t, m>; template<class t,class m> using slice = Scalar<t, m>; };
template<> struct base<2> { template<class t, class m> using type = Matrix<t, m>; template<class t,class m> using slice = Vector<t, m>; };
template<> struct base<3> { template<class t, class m> using type = Cube<t, m>;   template<class t,class m> using slice = Matrix<t, m>; };
template<> struct base<4> { template<class t, class m> using type = Tensor4<t, m>;   template<class t,class m> using slice = Cube<t, m>; };
template<> struct base<5> { template<class t, class m> using type = Tensor5<t, m>;   template<class t,class m> using slice = Tensor4<t, m>; };

template<int x, class a, class b> using rank2class = typename base<x>::template type<a,b>;

template<class a, class b> struct ranker<Scalar<a,b>> { static constexpr int value = 0; };
template<class a, class b> struct ranker<Vector<a,b>> { static constexpr int value = 1; };
template<class a, class b> struct ranker<Matrix<a,b>> { static constexpr int value = 2; };
template<class a, class b> struct ranker<Cube<a,b>>   { static constexpr int value = 3; };
template<class a, class b> struct ranker<Tensor4<a,b>>   { static constexpr int value = 4; };
template<class a, class b> struct ranker<Tensor5<a,b>>   { static constexpr int value = 5; };

template<class T> static constexpr int class2rank = ranker<T>::value;


template<class> struct determine_functor;
template<class> struct determine_evaluation;
template<class> struct determine_scalar;

template<class T> using _scalar = typename determine_scalar<T>::type;
template<class T> using _mathlib = typename MTF::tail<T>;
template<class T> using _ranker  = typename ranker<T>::type;
template<class T> using _functor = typename determine_functor<T>::type;
template<class T> using _evaluation = typename determine_evaluation<T>::type;
template<class T> static constexpr int _rankOf  = ranker<T>::value;

///DET FUNCTOR
template<template<class...> class tensor, class functor, class... set>
struct determine_functor<tensor<functor, set...>>{

	using derived = tensor<functor,set...>;
	using type = ifte<MTF::isPrimitive<functor>::conditional, Tensor_Core<derived>, functor>;
};

///DET EVALUATION
template<template<class...> class tensor, class functor, class... set>
struct determine_evaluation<tensor<functor, set...>>{
	using derived = tensor<functor,set...>;

	using type = ifte<MTF::isPrimitive<functor>::conditional || isCore<functor>::conditional,
			derived&,
			tensor<_scalar<derived>, _mathlib<derived>>>;
};

//SCALAR_TYPE----------------------------------------------------------------------------------------------
template<class T>
struct determine_scalar {
		using type = T;
};
template<template<class...> class tensor, class T, class... set>
struct determine_scalar<tensor<T, set...>> {
		using type = typename determine_scalar<T>::type;
};

}



#endif /* DETERMINERS_H_ */
