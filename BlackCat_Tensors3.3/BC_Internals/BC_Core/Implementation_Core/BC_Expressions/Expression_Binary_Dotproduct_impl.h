/*
 * Expression_Binary_Dotproduct_impl2.h
 *
 *  Created on: Jan 23, 2018
 *      Author: joseph
 */
#ifndef EXPRESSION_BINARY_DOTPRODUCT_IMPL2_H_
#define EXPRESSION_BINARY_DOTPRODUCT_IMPL2_H_

#include "../Determiners.h"

namespace BC {
//Required forward decs
template<class, class, class > class binary_expression_scalar_mul;
template<class, class > class unary_expression_transpose;
template<class > class Tensor_Core;
class mul;

template<class T> T&  cc(const T&  param) { return const_cast<T&> (param); }
template<class T> T&& cc(const T&& param) { return const_cast<T&&>(param); }
template<class T> T*  cc(const T*  param) { return const_cast<T*> (param); }

template<class> class front;
template<template<class...> class param, class first, class... set>
class front<param<first, set...>> {
	using type = first;
};

//DEFAULT TYPE
template<class> struct det_eval {
	static constexpr bool evaluate = true;
	static constexpr bool transposed = false;
	static constexpr bool scalar = false;
	template<class param> static auto getScalar(const param& p)-> _scalar<param>* { return nullptr; }
	template<class param> static auto getArray(const param& p) -> _scalar<param>* { throw std::invalid_argument("Attempting to use an array from an unevaluated context"); }
};

//IF TENSOR CORE (NON EXPRESSION)
template<class deriv> struct det_eval<Tensor_Core<deriv>> {
	static constexpr bool evaluate = false;
	static constexpr bool transposed = false;
	static constexpr bool scalar = false;

	template<class param> static _scalar<deriv>* getScalar(const param& p) { return nullptr; }
	template<class param> static _scalar<deriv>* getArray(const param& p) { return cc(p); }
};
////IF TRANSPOSE
template<class T, class deriv>
struct det_eval<unary_expression_transpose<T, Tensor_Core<deriv>>> {
	static constexpr bool evaluate = false;
	static constexpr bool transposed = true;
	static constexpr bool scalar = false;

	template<class param> static T* getScalar(const param& p) { return nullptr; }
	template<class param> static T* getArray(const param& p) { return cc(p.array); }
};


//IF A SCALAR BY TENSOR MUL OPERATION
template<class T, class d1, class d2>
struct det_eval<binary_expression_scalar_mul<T, Tensor_Core<d1>, Tensor_Core<d2>>> {
	static constexpr bool evaluate = false;
	static constexpr bool transposed = false;
	static constexpr bool scalar = true;

	template<class param>
	static constexpr bool left_scal = param::RANK() == 0;

	template<class param> static T* getScalar(const param& p) { return cc(left_scal<param> ? p.left : p.right); }
	template<class param> static T* getArray(const param& p)  { return cc(left_scal<param> ? p.right : p.left); }
};

//IF A SCALAR BY TENSOR MUL OPERATION R + TRANSPOSED
template<class T, class d1, class d2>
struct det_eval<binary_expression_scalar_mul<T, unary_expression_transpose<T, Tensor_Core<d1>>, Tensor_Core<d2>>> {
	static constexpr bool evaluate = false;
	static constexpr bool transposed = true;
	static constexpr bool scalar = true;

	template<class param> static T* getScalar(const param& p) { return cc(p.right); }
	template<class param> static T* getArray(const param& p) { return cc(p.left.array); }
};

//IF A SCALAR BY TENSOR MUL OPERATION L + TRANSPOSED
template<class T, class d1, class d2>
struct det_eval<binary_expression_scalar_mul<T, Tensor_Core<d1>, unary_expression_transpose<T, Tensor_Core<d2>>>> {
	static constexpr bool evaluate = false;
	static constexpr bool transposed = true;
	static constexpr bool scalar = true;

	template<class param> static T* getScalar(const param& p) { return cc(p.left.core()); }
	template<class param> static T* getArray(const param& p) { return cc(p.right.array.core()); }
};
}

#endif /* EXPRESSION_BINARY_DOTPRODUCT_IMPL2_H_ */
