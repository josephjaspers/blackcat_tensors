/*
 * Expression_Binary_Dotproduct_impl2.h
 *
 *  Created on: Jan 23, 2018
 *      Author: joseph
 */
#ifndef EXPRESSION_BINARY_DOTPRODUCT_IMPL2_H_
#define EXPRESSION_BINARY_DOTPRODUCT_IMPL2_H_

#include "BC_Utility/Determiners.h"

namespace BC {
//Required forward decs

template<class, class> class unary_expression;
template<class> class Core;
class transpose;
class scalar_mul;

template<class T> T&  cc(const T&  param) { return const_cast<T&> (param); }
template<class T> T&& cc(const T&& param) { return const_cast<T&&>(param); }
template<class T> T*  cc(const T*  param) { return const_cast<T*> (param); }

template<class> class front;
template<template<class...> class param, class first, class... set>
class front<param<first, set...>> {
	using type = first;
};

//DEFAULT TYPE
template<class T> struct det_eval {
	static constexpr bool evaluate = true;
	static constexpr bool transposed = false;
	static constexpr bool scalar = false;

	template<class param> static _scalar<param>* getScalar(const param& p) { return nullptr; }
	template<class param> static _scalar<param>* getArray(const param& p)  { throw std::invalid_argument("Attempting to use an array from an unevaluated context"); }
};
//
//IF TENSOR CORE (NON EXPRESSION)
template<class deriv> struct det_eval<Core<deriv>> {
	static constexpr bool evaluate = false;
	static constexpr bool transposed = false;
	static constexpr bool scalar = false;

	template<class param> static _scalar<deriv>* getScalar(const param& p) { return nullptr; }
	template<class param> static _scalar<deriv>* getArray(const param& p) { return cc(p); }
};
////IF TRANSPOSE
template<class deriv>
struct det_eval<unary_expression<Core<deriv>, transpose>> {
	static constexpr bool evaluate = false;
	static constexpr bool transposed = true;
	static constexpr bool scalar = false;

	template<class param> static _scalar<param>* getScalar(const param& p) { return nullptr; }
	template<class param> static _scalar<param>* getArray(const param& p) { return cc(p.array); }
};

//
////IF A SCALAR BY TENSOR MUL OPERATION
template<class d1, class d2>
struct det_eval<binary_expression<Core<d1>, Core<d2>, scalar_mul>> {
	using self = binary_expression<Core<d1>, Core<d2>, scalar_mul>;

	static constexpr bool evaluate = false;
	static constexpr bool transposed = false;
	static constexpr bool scalar = true;

	static constexpr bool left_scal = d1::DIMS() == 0;
	static constexpr bool right_scal = d2::DIMS() == 0;
	struct DISABLE;

	using left_scal_t  = std::conditional_t<left_scal,  self, DISABLE>;
	using right_scal_t = std::conditional_t<right_scal, self, DISABLE>;

	static _scalar<self>*  getArray(const left_scal_t& p) { return cc(p.right); }
	static _scalar<self>* getArray(const right_scal_t& p) { return cc(p.left);   }
	static _scalar<self>*  getScalar(const left_scal_t& p) { return cc(p.left); }
	static _scalar<self>* getScalar(const right_scal_t& p) { return cc(p.right); }

};
//
//IF A SCALAR BY TENSOR MUL OPERATION R + TRANSPOSED
template<class d1, class d2>
struct det_eval<binary_expression<unary_expression<Core<d1>, transpose>, Core<d2>, scalar_mul>> {
	static constexpr bool evaluate = false;
	static constexpr bool transposed = true;
	static constexpr bool scalar = true;

	template<class param> static _scalar<param>* getScalar(const param& p) { return cc(p.right); }
	template<class param> static _scalar<param>* getArray(const param& p) { return cc(p.left.array); }
};

//IF A SCALAR BY TENSOR MUL OPERATION L + TRANSPOSED
template<class d1, class d2>
struct det_eval<binary_expression<Core<d1>, unary_expression<Core<d2>, transpose>, scalar_mul>> {
	static constexpr bool evaluate = false;
	static constexpr bool transposed = true;
	static constexpr bool scalar = true;

	template<class param> static _scalar<param>* getScalar(const param& p) { return cc(p.left.getIterator()); }
	template<class param> static _scalar<param>* getArray(const param& p) { return cc(p.right.array.getIterator()); }
};
}

#endif /* EXPRESSION_BINARY_DOTPRODUCT_IMPL2_H_ */
