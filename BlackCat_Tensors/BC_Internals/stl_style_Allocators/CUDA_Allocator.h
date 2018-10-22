/*
 * CUDA_Allocator.h
 *
 *  Created on: Oct 22, 2018
 *      Author: joseph
 */


#ifdef __CUDACC__
#ifndef CUDA_ALLOCATOR_H_
#define CUDA_ALLOCATOR_H_



namespace BC {

class GPU;

namespace module {
namespace stl {

template<class T>
struct CUDA_Allocator : GPU {

	template<typename T>
	static T*& allocate(T*& t, int sz=1) {
		cudaMalloc((void**) &t, sizeof(T) * sz);
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



};

}
}
}



#endif /* CUDA_ALLOCATOR_H_ */
#endif
