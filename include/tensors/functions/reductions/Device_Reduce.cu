/*
 * Device_Add.cu
 *
 *  Created on: Aug 27, 2019
 *      Author: joseph
 */

#ifdef __CUDACC__
#ifndef BLACKCATTENSORS_TENSORS_FUNCTIONS_REDUCTIONS_DEVICE_REDUCE_CU_
#define BLACKCATTENSORS_TENSORS_FUNCTIONS_REDUCTIONS_DEVICE_REDUCE_CU_

#include <cuda_runtime_api.h>
#include <cuda.h>

//Reference:
//https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//-- Modified by Joseph Jaspers original kernel (predominantlywriter Mark Harris

namespace BC {
namespace tensors {
namespace functions {

template <unsigned int blockSize, class T> __device__
void warpReduce(volatile T *sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize, class Expression, class Scalar, class T> __global__
void reduce6(Expression g_idata, T *buffer, Scalar scalar_out, unsigned int n) {
	extern __shared__ T sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	while (i < n) {
		sdata[tid] += g_idata[i] + g_idata[i + blockSize];
		i += gridSize;
	}

	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	if (tid == 0) buffer[blockIdx.x] = sdata[0];

	//accumulate to the final output
	__syncthreads();
	if (tid == 0) {
		scalar_out[0] = 0;
		for (int i = 0; i < (n/128); ++i) {
			scalar_out[0] += buffer[i];
		}
	}
}

template<class Stream, class ScalarOutput,  class Expression>
static void sum(Stream stream, ScalarOutput output, Expression expression) {

	using value_type = typename Expression::value_type;

	BC::size_t n = expression.size();
	BC::size_t blocks = BC::blocks(n);	///num blocks the kernel will be called
	BC::size_t threads = BC::threads(n); ///num threads the kernel will be called with
	threads = threads > n ? std::ceil(n/32) * 32 : threads;
	BC::size_t smemSize = threads * sizeof(typename Expression::value_type);
	BC::size_t buffer_size = blocks;

	//TODO remove the reserve call
	stream.template get_allocator_rebound<value_type>().reserve(buffer_size);
	value_type* buffer = stream.template get_allocator_rebound<value_type>().allocate(buffer_size);

	switch (threads)
	{
		case 512:
		reduce6<512><<< blocks, threads, smemSize >>>(expression, buffer, output, n); break;
		case 256:
		reduce6<256><<< blocks, threads, smemSize >>>(expression, buffer, output, n); break;
		case 128:
		reduce6<128><<< blocks, threads, smemSize >>>(expression, buffer, output, n); break;
		case 64:
		reduce6< 64><<< blocks, threads, smemSize >>>(expression, buffer, output, n); break;
		case 32:
		reduce6< 32><<< blocks, threads, smemSize >>>(expression, buffer, output, n); break;
		case 16:
		reduce6< 16><<< blocks, threads, smemSize >>>(expression, buffer, output, n); break;
		case 8:
		reduce6< 8><<< blocks, threads, smemSize >>>(expression, buffer, output, n); break;
		case 4:
		reduce6< 4><<< blocks, threads, smemSize >>>(expression, buffer, output, n); break;
		case 2:
		reduce6< 2><<< blocks, threads, smemSize >>>(expression, buffer, output, n); break;
		case 1:
		reduce6< 1><<< blocks, threads, smemSize >>>(expression, buffer, output, n); break;
	}

	stream.template get_allocator_rebound<value_type>().deallocate(buffer, buffer_size);

}

}
}
}

#endif /* DEVICE_ADD_CU_ */
#endif //ifdef __CUDACC__
