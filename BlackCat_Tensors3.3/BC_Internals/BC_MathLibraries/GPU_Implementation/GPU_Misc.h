/*
 * GPU_Misc.h
 *
 *  Created on: May 6, 2018
 *      Author: joseph
 */

#ifndef GPU_MISC_H_
#define GPU_MISC_H_

namespace BC {


template<class core_lib>
struct GPU_Misc {

	static int blocks(int sz) { return core_lib::blocks(sz); }
	static int threads() { return core_lib::threads(); }


	template<typename T, typename J>
	static void fill(T t, const J j) {
		gpu_impl::fill<<<blocks(t.size()),threads()>>>(t, j);
		cudaDeviceSynchronize();
	}

	template<template<class...> class T, class...set, class Value>
	static void fill(T<set...> t, Value val) {
		gpu_impl::fill<<<blocks(t.size()),threads()>>>(t, val);
		cudaDeviceSynchronize();
	}

	template<typename T>
	static void zero(T t) {
		gpu_impl::fill<<<blocks(t.size()),threads()>>>(t, 0);
		cudaDeviceSynchronize();
	}
	template<template<class...> class T, class...set>
	static void zero(T<set...> t, int sz) {
		gpu_impl::fill<<<blocks(t.size()),threads()>>>(t, 0);
		cudaDeviceSynchronize();
	}

	template<typename T>
	static void randomize(T t, float lower_bound, float upper_bound) {
		gpu_impl::randomize<<<blocks(t.size()),threads()>>>(t, lower_bound, upper_bound, rand());
		cudaDeviceSynchronize();
	}
	template<template<class...> class T, class...set>
	static void randomize(T<set...> t, float lower_bound, float upper_bound) {
		gpu_impl::randomize<<<blocks(t.size()),threads()>>>(t, lower_bound, upper_bound, rand());
		cudaDeviceSynchronize();
	}


};

}



#endif /* GPU_MISC_H_ */
