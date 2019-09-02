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
namespace exprs {
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

template <class T, class Expression> __global__
void small_sum(T output, Expression array, BC::size_t n) {
	if (threadIdx.x==0) {
		output[0] = 0;
		for (int i = 0; i < n; ++i) {
			output[0] += array[i];
		}
	}
}


template <unsigned int blockSize, class Expression, class Scalar, class T> __global__
void reduce6(Expression g_idata, T *buffer, Scalar scalar_out, unsigned int n, unsigned buffer_size) {
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
		for (int i = 0; i < (buffer_size); ++i) {
			scalar_out[0] += buffer[i];
		}
	}
}

template<class>
struct Reduce;

template<>
struct Reduce<BC::device_tag> {

	template<class Stream, class ScalarOutput,  class Expression>
	static void sum(Stream stream, ScalarOutput output, Expression expression) {
		/*
		 *  It is required that we allocate all 'buffer' (or workspace) memory before enqueue-ing the function.
		 *  Functions that are enqueued are expected to accept all workspace memory required as an argument
		 *  (not allocating them from within the function).
		 *
		 *  Thats because our 'logging_stream' keeps tracks of all allocations/deallocations but ignores enqueued functions
		 *  That way we can use a single pass of the expression to log all memory usage and a second pass of the expression
		 *  (first allocating the max amount of memory required) to ensure we only need 1 actual allocation per expression.
		 *
		 *  IE  y.alias() = w * x + sum(x)
		 *  -- despite having 2 temporaries will at most will call cudaMalloc once.
		 *  This ensures that we limit the maximum number of allocations on the gpu to 1 per expression.
		 *  If the sizeof the reserved memory is equal or exceeds the max amount of memory, obviously we don't have to have
		 *  any cudaMalloc calls. (but we wouldn't be able to know this for sure without the first pass with the logging-stream)
		 *
		 *  This to explain why 'sum' is broken into two oddly separate functions. (sum and sum_implementation)
		 *
		 */

		if (expression.size() > 32) {
			using value_type = typename Expression::value_type;
			BC::size_t buffer_size = BC::calculate_block_dim(expression.size());	///num blocks the kernel will be called
			value_type* buffer = stream.template get_allocator_rebound<value_type>().allocate(buffer_size);
			stream.enqueue([&]() { sum_implementation(stream, output, expression, buffer); });
			stream.template get_allocator_rebound<value_type>().deallocate(buffer, buffer_size);
		} else {
			stream.enqueue([&]() {
				small_sum <<< 1, 1, 0, stream>>> (output, expression, expression.size());
			});
		}
	}


private:
	template<class Stream, class ScalarOutput,  class Expression>
	static void sum_implementation(
			Stream stream,
			ScalarOutput output,
			Expression expression,
			typename Expression::value_type* buffer) {

		using value_type = typename Expression::value_type;
		BC::size_t n = expression.size();
		BC::size_t blocks = BC::calculate_block_dim(n);	///num blocks the kernel will be called
		BC::size_t threads = BC::calculate_threads(n); ///num threads the kernel will be called with
		BC::size_t smemSize = threads * sizeof(typename Expression::value_type);

		switch (threads)
		{
			case 512:
			reduce6<512><<< blocks, threads, smemSize, stream>>>(expression, buffer, output, n, blocks); break;
			case 256:
			reduce6<256><<< blocks, threads, smemSize, stream>>>(expression, buffer, output, n,blocks); break;
			case 128:
			reduce6<128><<< blocks, threads, smemSize, stream>>>(expression, buffer, output, n,blocks ); break;
			case 64:
			reduce6< 64><<< blocks, threads, smemSize, stream>>>(expression, buffer, output, n,blocks); break;
			case 32:
			reduce6< 32><<< blocks, threads, smemSize, stream>>>(expression, buffer, output, n,blocks); break;
			case 16:
			reduce6< 16><<< blocks, threads, smemSize, stream>>>(expression, buffer, output, n,blocks); break;
			case 8:
			reduce6< 8><<< blocks, threads, smemSize, stream>>>(expression, buffer, output, n,blocks); break;
			case 4:
			reduce6< 4><<< blocks, threads, smemSize, stream>>>(expression, buffer, output, n,blocks); break;
			case 2:
			reduce6< 2><<< blocks, threads, smemSize, stream>>>(expression, buffer, output, n,blocks); break;
			case 1:
			reduce6< 1><<< blocks, threads, smemSize, stream>>>(expression, buffer, output, n,blocks); break;
		}
	}
};

}
}
}
}

#endif /* DEVICE_ADD_CU_ */
#endif //ifdef __CUDACC__
