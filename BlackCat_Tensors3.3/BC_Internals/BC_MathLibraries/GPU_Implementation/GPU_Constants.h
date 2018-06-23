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
	static float* static_initialize(float value) {
		float* t;
		cudaMallocManaged((void**) &t, sizeof(float));
		cudaMemcpy(t, &value, sizeof(float), cudaMemcpyHostToDevice);
		return t;
	}

	static void scalar_mul(float* eval, float* a, float* b) {

//		float a_;
//		float b_;
//
//		core_lib::DeviceToHost(&a_, a);
//		core_lib::DeviceToHost(&b_, b);
//
//		std::cout << " a = " << a_ << std::endl;
//		std::cout << " b = " << b_ << std::endl;
//
//
//		std::cout << " scalar mul gpu  " << std::endl;
		gpu_impl::scalar_mul<<<1, 1>>>(eval, a, b);
		cudaDeviceSynchronize();
//
//
//
//		core_lib::DeviceToHost(&a_, eval);
//		std::cout << " output = " << a_ << std::endl;
//


	}


};

}



#endif /* GPU_CONSTANTS_H_ */
