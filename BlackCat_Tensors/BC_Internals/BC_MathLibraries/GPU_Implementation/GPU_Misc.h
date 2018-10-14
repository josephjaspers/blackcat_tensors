/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef GPU_MISC_H_
#define GPU_MISC_H_

namespace BC {


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


	struct BC_curand_handle {
		curandState* rand_ptr;
		static constexpr int floating_point_decimal_length = 10000;

		BC_curand_handle() {
			cudaMalloc(&rand_ptr, sizeof(curandState));
			gpu_impl::init_curand_handle<<<1,1>>>(rand_ptr);
		}

		template<class scalar_t> 	__device__
		scalar_t operator () (scalar_t lower_bound, scalar_t upper_bound) const {
			scalar_t value = curand(&*rand_ptr) % floating_point_decimal_length;
			value /= floating_point_decimal_length;
			value *= (upper_bound - lower_bound);
			value += lower_bound;
			return value;
		}
	} static RAND_HANDLE_PRIMARY;

	template<class scalar_t>
	struct rand_t {

		using rand_handle_t = BC_curand_handle;

		scalar_t min;
		scalar_t max;

		rand_handle_t rand_handle_obj;

		rand_t(scalar_t min_, scalar_t max_) : min(min_), max(max_),
				rand_handle_obj(RAND_HANDLE_PRIMARY) {}

		__device__
		auto operator () (scalar_t v) const {
			return rand_handle_obj(min,max);
		}
		__host__
		scalar_t operator () (scalar_t& v) const {
			return 0;
		}
	};


};// = gpu_impl::BC_curand_handle(

}



#endif /* GPU_MISC_H_ */
