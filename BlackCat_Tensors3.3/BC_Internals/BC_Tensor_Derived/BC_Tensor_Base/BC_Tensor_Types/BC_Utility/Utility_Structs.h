/*
 * Internal_Shapes.h
 *
 *  Created on: Apr 15, 2018
 *      Author: joseph
 */

#ifndef INTERNAL_SHAPES_H_
#define INTERNAL_SHAPES_H_

#include <vector>
#include <initializer_list>
#include <iostream>
#include "../BlackCat_Internal_Definitions.h"
/*
 * Defines stack array and lambda array
 * These are just two trivially classes used for storing small amounts of homogenous data types, generally ints
 */

namespace BC {

//-------------------------------Lightweight array, implemented as a homogeneous tuple-------------------------------//
	template<class T, int size_>
	struct stack_array : stack_array<T, size_ - 1> {
		T value;

		template<class... values>
		__BCinline__ stack_array(T val, values... integers) : stack_array<T, size_ - 1>(integers...), value(val) {}
		__BCinline__ stack_array() {}

		__BCinline__ const T& operator [] (int i) const { return (&value)[-i]; }
		__BCinline__ 	   T& operator [] (int i) 		{ return (&value)[-i]; }
		__BCinline__ static constexpr int size() 		{ return size_; }
	};

	//---------------------base_case----------------------//
	template<class T> struct stack_array<T, 0> {};


	//---------------------short_hand----------------------//
	template<class T, class... vals> __BChd__
		auto array(T front, vals... values) { return stack_array<T, sizeof...(values) + 1>(front, values...); }


//-------------------------------Lightweight lambda-wrapper to enable usage of the bracket-operator-------------------------------//
	template<class ref>
	struct lambda_array{
		ref value;
		__BCinline__ lambda_array(ref a) : value(a) {}

		__BCinline__ const auto operator [](int i) const { return value(i); }
		__BCinline__ 	   auto operator [](int i) 		 { return value(i); }
	};

	//accepts a lambda that takes a single-integer parameter
	template<class T> __BChd__ auto l_array(T data) { return lambda_array<T>(data); }
}


#endif /* INTERNAL_SHAPES_H_ */
