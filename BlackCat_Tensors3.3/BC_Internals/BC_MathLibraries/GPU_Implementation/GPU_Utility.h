/*
 * GPU_Utility.h
 *
 *  Created on: May 6, 2018
 *      Author: joseph
 */

#ifndef GPU_UTILITY_H_
#define GPU_UTILITY_H_


namespace BC {

template<class core_lib>
struct GPU_Utility {


	template<typename T>
	static T*& initialize(T*& t, int sz=1) {
		cudaMalloc((void**) &t, sizeof(T) * sz);
		return t;
	}
	template<typename T>
	static T*& zero_initialize(T*& t, int sz=1) {
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
	static T*& unified_initialize(T*& t, int sz) {
		cudaMallocManaged((void**) &t, sizeof(T) * sz);
		return t;
	}

	template<typename T> __host__ __device__
	static void destroy(T* t) {
		cudaFree((void*)t);
	}
	template<typename T>
	static void destroy(T t) {
		throw std::invalid_argument("destruction on class object");
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

		BC::print(print, ranks, outer, order, print_length);
		delete[] print;
	}
	template<class RANKS, class os>
	static void printSparse(const float* ary, const RANKS ranks, const os outer, int order, int print_length) {
		int sz = calc_size(ranks, order);
		float* print = new float[sz];
		DeviceToHost(print, ary, sz);

		BC::printSparse(print, ranks, outer, order, print_length);
		delete[] print;
	}


};


}


#endif /* GPU_UTILITY_H_ */
