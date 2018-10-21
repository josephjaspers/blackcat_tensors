/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef GPU_UTILITY_H_
#define GPU_UTILITY_H_


namespace BC {

struct CUDA_Allocator {

	template<typename T>
	static T*& allocate(T*& t, int sz=1) {
		cudaMalloc((void**) &t, sizeof(T) * sz);
		return t;
	}
	template<typename T>
	static T*& zero_allocate(T*& t, int sz=1) {
		cudaMalloc((void**) &t, sizeof(T) * sz);
		cudaMemset((void**) &t, 0, sizeof(T) * sz);
		return t;
	}

	static void barrier() {
		cudaDeviceSynchronize();
	}


	template<class T>
	static void HostToDevice(T* t, const T* u, int size = 1) {
		cudaDeviceSynchronize();
		cudaMemcpy(t, u, sizeof(T) * size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}
	template<class T>
	static void DeviceToHost(T* t, const T* u, int size = 1) {
		cudaDeviceSynchronize();
		cudaMemcpy(t, u, sizeof(T) * size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}

	template<typename T>
	static T*& unified_allocate(T*& t, int sz) {
		cudaMallocManaged((void**) &t, sizeof(T) * sz);
		return t;
	}

	template<typename T>
	static void deallocate(T* t) {
		cudaFree((void*)t);
	}
	template<typename T>
	static void deallocate(T t) {
		//empty
	}


	template<class ranks>
	static int calc_size(ranks R, int order) {
		if (order == 0) {
			return 1;
		}

		int sz = 1;
		for (int i = 0; i < order; ++i) {
			sz *= R[i];
		}
		return sz;
	}

	template<class RANKS, class os>
	static void print(const float* ary, const RANKS ranks,const os outer, int order, int print_length) {
		int sz = calc_size(ranks, order);
		float* print = new float[sz];

		DeviceToHost(print, ary, sz);

		BC::IO::print(print, ranks, outer, order, print_length);
		delete[] print;
	}
	template<class RANKS, class os>
	static void printSparse(const float* ary, const RANKS ranks, const os outer, int order, int print_length) {
		int sz = calc_size(ranks, order);
		float* print = new float[sz];
		DeviceToHost(print, ary, sz);

		BC::IO::printSparse(print, ranks, outer, order, print_length);
		delete[] print;
	}


};


}


#endif /* GPU_UTILITY_H_ */
