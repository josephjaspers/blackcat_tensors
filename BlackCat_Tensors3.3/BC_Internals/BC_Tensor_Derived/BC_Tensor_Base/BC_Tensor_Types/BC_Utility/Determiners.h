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

template<class> class Core;
template<class> class Tensor_Slice;
template<class> class Tensor_Scalar;
template<class> class Tensor_Reshape;
template<class,class> class Scalar;
template<class,class> class Vector;
template<class,class> class Matrix;
template<class,class> class Cube;
template<class,class> class Tensor4;
template<class,class> class Tensor5;
template<class...> class DISABLED_LIST;
struct DISABLED;
class BC_Type;

//determines if a functor is primary core
template<class T> struct isPrimaryCore { static constexpr bool conditional = !std::is_base_of<BC_Type, T>::value; };
template<class T> struct isPrimaryCore<Core<T>> { static constexpr bool conditional = true; };
template<class T> static constexpr bool pCore_b = isPrimaryCore<T>::conditional;

//determines the tensor type from an integer corresponding to its dimensionality
template<int> struct tensor_of { template<class t, class m> using type = DISABLED;	template<class t, class m> using slice = DISABLED; };
template<> struct tensor_of<0> { template<class t, class m> using type = Scalar<t,m>;  	template<class t, class m> using slice = DISABLED; };
template<> struct tensor_of<1> { template<class t, class m> using type = Vector<t, m>; 	template<class t,class m> using slice = Scalar<t, m>; };
template<> struct tensor_of<2> { template<class t, class m> using type = Matrix<t, m>; 	template<class t,class m> using slice = Vector<t, m>; };
template<> struct tensor_of<3> { template<class t, class m> using type = Cube<t, m>;   	template<class t,class m> using slice = Matrix<t, m>; };
template<> struct tensor_of<4> { template<class t, class m> using type = Tensor4<t, m>;  template<class t,class m> using slice = Cube<t, m>; };
template<> struct tensor_of<5> { template<class t, class m> using type = Tensor5<t, m>;  template<class t,class m> using slice = Tensor4<t, m>; };
template<int x, class a, class b> using tensor_of_t = typename tensor_of<x>::template type<a,b>;

//determines the dimensionality of a tensor
template<class> struct dimension_of;
template<class a, class b> struct dimension_of<Scalar<a,b>> { static constexpr int value = 0; using ml = b;};
template<class a, class b> struct dimension_of<Vector<a,b>> { static constexpr int value = 1; using ml = b;};
template<class a, class b> struct dimension_of<Matrix<a,b>> { static constexpr int value = 2; using ml = b;};
template<class a, class b> struct dimension_of<Cube<a,b>>   { static constexpr int value = 3; using ml = b;};
template<class a, class b> struct dimension_of<Tensor4<a,b>>   { static constexpr int value = 4; using ml = b;};
template<class a, class b> struct dimension_of<Tensor5<a,b>>   { static constexpr int value = 5; using ml = b;};
template<class T> static constexpr int dimension_of_v = dimension_of<T>::value;

//determines if the type is a BC tensor
template<class> struct isTensor {static constexpr bool conditional = false; };
template<class a, class b> struct isTensor<Scalar<a,b>> { static constexpr bool conditional = true; };
template<class a, class b> struct isTensor<Vector<a,b>> { static constexpr bool conditional = true; };
template<class a, class b> struct isTensor<Matrix<a,b>> { static constexpr bool conditional = true; };
template<class a, class b> struct isTensor<Cube<a,b>>   { static constexpr bool conditional = true; };
template<class a, class b> struct isTensor<Tensor4<a,b>>   { static constexpr bool conditional = true; };
template<class a, class b> struct isTensor<Tensor5<a,b>>   { static constexpr bool conditional = true; };
template<class T> static constexpr bool isTensor_b = isTensor<T>::conditional;

//forward declerations (implementations bottom)
template<class> struct determine_functor;
template<class> struct determine_evaluation;
template<class> struct determine_scalar;
template<class> struct determine_iterator;
template<class> struct determine_mathlibrary;
template<class> struct determine_tensor_scalar;

//Shorthand for the metatemplate functions these are used VERY COMMONLY
template<class T> using _scalar = typename determine_scalar<T>::type;
template<class T> using _mathlib = typename determine_mathlibrary<T>::type;
template<class T> using _ranker  = typename dimension_of<T>::type;
template<class T> using _functor = typename determine_functor<T>::type;
template<class T> using _evaluation = T; // fix me
template<class T> static constexpr int _dimension_of  = dimension_of<T>::value;
template<class T> using _iterator = typename determine_iterator<T>::type;
template<class T> using _tensor_scalar = typename determine_tensor_scalar<T>::type;

///DET FUNCTOR----------------------------------------------------------------------------------------------
template<template<class...> class tensor, class functor, class... set>
struct determine_functor<tensor<functor, set...>>{

	using derived = tensor<functor,set...>;
	using type = ifte<std::is_base_of<BC_Type,functor>::value, functor, Core<derived>>;
};

//SCALAR_TYPE----------------------------------------------------------------------------------------------
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



#endif /* DETERMINERS_H_ */
