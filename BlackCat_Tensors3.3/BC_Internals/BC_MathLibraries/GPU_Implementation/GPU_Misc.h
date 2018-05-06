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
	static void fill(T t, const J j, int sz) {
		gpu_impl::fill<<<blocks(sz),threads()>>>(t, j, sz);
		cudaDeviceSynchronize();
	}

	template<typename T, typename J>
	static void fill(T t, const J* j, int sz) {
		gpu_impl::fill<<<blocks(sz),threads()>>>(t, j, sz);
		cudaDeviceSynchronize();
	}
	template<typename T>
	static void zero(T t, int sz) {
		gpu_impl::zero<<<blocks(sz),threads()>>>(t, sz);
		cudaDeviceSynchronize();
	}
	template<template<class...> class T, class...set>
	static void zero(T<set...> t, int sz) {
		gpu_impl::zero<<<blocks(sz),threads()>>>(t, sz);
		cudaDeviceSynchronize();
	}

	template<typename T, class J>
	static void randomize(T t, J lower_bound, J upper_bound, int sz) {
		gpu_impl::randomize<<<blocks(sz),threads()>>>(t, lower_bound, upper_bound, sz, rand());
		cudaDeviceSynchronize();
	}


};

}



#endif /* GPU_MISC_H_ */
