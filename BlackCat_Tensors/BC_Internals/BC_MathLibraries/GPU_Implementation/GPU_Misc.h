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
