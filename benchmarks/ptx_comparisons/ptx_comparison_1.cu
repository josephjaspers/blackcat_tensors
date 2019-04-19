/*
 * ptx_comparison.h
 *
 *  Created on: Apr 19, 2019
 *      Author: joseph
 */

#ifndef PTX_COMPARISON_H_
#define PTX_COMPARISON_H_

#include "../../include/BlackCat_Tensors.h"

template<class T> __global__
void cuda_cwise_test_kernel(T* out, int length, T* a, T* b, T* c, T* d) {

		//grid stride loop
	    auto i = blockIdx.x * blockDim.x + threadIdx.x;
	    for (; i < length; i += blockDim.x * gridDim.x) {
	        out[i] = a[i] + b[i]- c[i] / d[i];
	    }
}


int test() {
	BC::Matrix<float, BC::Allocator<BC::device_tag, float>> m;
	m = m + m - m / m;

	float* mptr = m.memptr();
	cuda_cwise_test_kernel<<<BC::threads(), BC::blocks(m.size())>>>(
			mptr, m.size(),
			mptr,mptr,mptr,mptr);

	return 0;
}



#endif /* PTX_COMPARISON_H_ */
