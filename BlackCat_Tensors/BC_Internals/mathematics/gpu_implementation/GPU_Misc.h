/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef GPU_MISC_H_
#define GPU_MISC_H_

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

namespace BC {

//TODO deprecate this, move towards using thrust library
template<class core_lib>
class GPU_Misc {

    static int blocks(int sz) { return core_lib::blocks(sz); }
    static int threads() { return core_lib::threads(); }
public:
    template<typename T>
    static void randomize(T t, float lower_bound, float upper_bound) {
        gpu_impl::randomize<<<blocks(t.size()),threads()>>>(t, lower_bound, upper_bound, std::rand());
        cudaDeviceSynchronize();
    }
    template<template<class...> class T, class...set>
    static void randomize(T<set...> t, float lower_bound, float upper_bound) {
        gpu_impl::randomize<<<blocks(t.size()),threads()>>>(t, lower_bound, upper_bound, std::rand());
        cudaDeviceSynchronize();
    }

   template<class T>
   struct rand_handle {

    	mutable thrust::minstd_rand rng;
    	mutable thrust::uniform_real_distribution<float> dist;
    	unsigned index = 1;

		   __host__ __device__
	    rand_handle(float lb, float ub) : dist(lb, ub) {}

		   __host__ __device__
	   float operator () () const {
		   return dist(rng);
	   }

		   __host__ __device__
	    void operator () (float& val) const {
		   val = dist(rng);
	   }

	   __host__ __device__
	   float operator [] (unsigned i) const {
		   return  dist(rng);
	   }

   };

	static auto make_rand_gen(float lower, float upper) {
		return rand_handle<float>(lower, upper);
	}

};// = gpu_impl::BC_curand_handle(

}



#endif /* GPU_MISC_H_ */
