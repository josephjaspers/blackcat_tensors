/*
 * CUda_Managed_Allocator.h
 *
 *  Created on: Oct 24, 2018
 *      Author: joseph
 */
#ifdef __CUDACC__
#ifndef CUDA_MANAGED_ALLOCATOR_H_
#define CUDA_MANAGED_ALLOCATOR_H_



namespace BC {
namespace module {
class GPU;

namespace stl {

template<class T>
struct CUDA_Managed_Allocator : GPU {

	template<typename T>
	static T*& allocate(T*& t, int sz=1) {
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
};
}
}
}





#endif /* CUDA_MANAGED_ALLOCATOR_H_ */
#endif
