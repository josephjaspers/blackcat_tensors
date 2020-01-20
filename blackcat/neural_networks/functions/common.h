/*
 * Caffe_Cuda.h
 *
 *  Created on: Nov 10, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_NEURALNETWORKS_FUNCTIONS_COMMON_CUDA_H_
#define BLACKCAT_TENSORS_NEURALNETWORKS_FUNCTIONS_COMMON_CUDA_H_

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif
namespace bc {
namespace caffe {

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

}
}

#endif /* CAFFE_CUDA_H_ */
