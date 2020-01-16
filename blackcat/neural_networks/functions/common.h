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

#ifdef __CUDACC__

#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
		i < (n); \
		i += blockDim.x * gridDim.x)


// CUDA: use 512 threads per block
static constexpr int CAFFE_CUDA_NUM_THREADS = 512;
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

//This is actually the same way BC calculates blocks
// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}
#endif


}
}

#endif /* CAFFE_CUDA_H_ */
