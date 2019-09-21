/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_CORE_STRUCTURES_H_
#define BC_CORE_STRUCTURES_H_

#include "BlackCat_Common.h"
#include "utility/Utility.h"

/*
 * Defines stack array and lambda array
 * These are just two trivially classes used for storing small amounts of homogenous internal types, generally ints
 * this array class is used instead of std::array as it must support usability in CUDA kernels  
 */

namespace BC {
namespace utility {
//-------------------------------Lightweight array -------------------------------//
    template<int size_, class T>
    struct array {
    	static constexpr int tensor_dimension = size_;

        T value[size_] = { 0 } ;

        BCINLINE const T& operator [] (int i) const { return value[i]; }
        BCINLINE        T& operator [] (int i)      { return value[i]; }
        BCINLINE static constexpr BC::size_t size() { return size_; }
    };

    //---------------------base_case----------------------//
    template<class T> struct array<0, T> {};


    //---------------------short_hand----------------------//
    template<class T, class... vals> BCINLINE
    auto make_array(T front, vals... values) {
    	return array<sizeof...(values) + 1, T> {front, values...};
    }

	//make array but specify the type 
	template<class T, class... vals> BCINLINE
		auto make_array_t(vals... values) {
		return array<sizeof...(values), T> {values...};
	}


//-------------------------------Lightweight lambda-wrapper to enable usage of the bracket-operator-------------------------------//
    template<int dimension, class scalar, class ref>
    struct lambda_array{
    	static constexpr int tensor_dimension = dimension;
        ref value;
        BCINLINE lambda_array(ref a) : value(a) {}

        BCINLINE const scalar operator [](int i) const { return value(i); }
        BCINLINE       scalar operator [](int i)       { return value(i); }
    };

    //accepts a lambda that takes a single-integer parameter
    template<int dimension, class func> BCHOSTDEV
    auto make_lambda_array(func internal) {
    	return lambda_array<dimension, decltype(internal(0)), func>(internal);
    }
}
}

#endif /* INTERNAL_SHAPES_H_ */
