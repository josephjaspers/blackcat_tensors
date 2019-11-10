/*
 * Caffe_Cuda.h
 *
 *  Created on: Nov 10, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_CAFFE_CAFFE_CUDA_H_
#define BLACKCAT_TENSORS_CAFFE_CAFFE_CUDA_H_


namespace BC {
namespace caffe {


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

}
}

#endif /* CAFFE_CUDA_H_ */
