/*
 * GPU_Constants.h
 *
 *  Created on: Jun 10, 2018
 *      Author: joseph
 */

#ifndef GPU_CONSTANTS_H_
#define GPU_CONSTANTS_H_

namespace BC {

template<class core_lib>
struct GPU_Constants {
	static float* static_initialize(int sz, float value) {
		float* t;
		cudaMallocManaged((void**) &t, sizeof(float));
		cudaMemcpy(t, &value, sizeof(float), cudaMemcpyHostToDevice);
		return t;
	}

	template<class U, class T, class V>
	static void scalar_mul(U eval, T a, V b) {
		gpu_impl::scalar_mul<<<1, 1>>>(eval, a, b);
		cudaDeviceSynchronize();
	}


};

}



#endif /* GPU_CONSTANTS_H_ */
