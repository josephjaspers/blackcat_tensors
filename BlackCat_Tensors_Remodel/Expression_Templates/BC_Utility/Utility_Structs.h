/*
 * Internal_Shapes.h
 *
 *  Created on: Apr 15, 2018
 *      Author: joseph
 */

#ifndef INTERNAL_SHAPES_H_
#define INTERNAL_SHAPES_H_

#include "../BlackCat_Internal_Definitions.h"
/*
 * Defines stack array and lambda array
 * These are just two trivially classes used for storing small amounts of homogenous internal types, generally ints
 */

namespace BC {

//-------------------------------Lightweight array, implemented as a homogeneous tuple-------------------------------//
	template<int size_, class T>
	struct stack_array : stack_array< size_ - 1, T> {
		T value;

		template<class... values>
		__BCinline__ stack_array(T val, values... integers) : stack_array<size_ - 1, T>(integers...), value(val) {}
		__BCinline__ stack_array() {}

		__BCinline__ const T& operator [] (int i) const { return (&value)[-i]; }
		__BCinline__ 	   T& operator [] (int i) 		{ return (&value)[-i]; }
		__BCinline__ static constexpr int size() 		{ return size_; }
	};

	//---------------------base_case----------------------//
	template<class T> struct stack_array<0, T> {};


	//---------------------short_hand----------------------//
	template<class T, class... vals> __BChd__
		auto array(T front, vals... values) { return stack_array<sizeof...(values) + 1, T>(front, values...); }


//-------------------------------Lightweight lambda-wrapper to enable usage of the bracket-operator-------------------------------//
	template<int dimension, class scalar, class ref>
	struct lambda_array{
		ref value;
		__BCinline__ lambda_array(ref a) : value(a) {}

		__BCinline__ const scalar operator [](int i) const { return value(i); }
		__BCinline__ 	   scalar operator [](int i) 		 { return value(i); }
	};

	//accepts a lambda that takes a single-integer parameter
	template<int dimension, class func> __BChd__ auto l_array(func internal) { return lambda_array<dimension, decltype(internal(0)), func>(internal); }


	//-------------------------------stupid wrapper around pointer than use a template<int> to remember its length -------------------------------//
	template<int x, class scalar_t>
	struct pointer_array {
		scalar_t* array;
		__BCinline__ pointer_array(scalar_t* array_) : array(array_) {}

		__BCinline__ const scalar_t& operator [](int i) const { return array[i]; }
		__BCinline__ 	   scalar_t& operator [](int i) 	{ return array[i]; }
	};

	template<int x, class scalar> __BChd__ auto ptr_array(scalar* array) {
		return pointer_array<x, scalar>(array);
	}
}


#endif /* INTERNAL_SHAPES_H_ */
