/*
 * Device.h
 *
 *  Created on: Dec 3, 2018
 *      Author: joseph
 */


#ifdef __CUDACC__
#ifndef DEVICE_H_
#define DEVICE_H_

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../common.h"

namespace bc {

struct device_tag;

namespace random {
namespace device_impl {

__global__
static void bc_curand_init(curandState_t* state) {

	int i = blockIdx.x + blockDim.x + threadIdx.x;
	if (!state && i==0)
		curand_init(0,0,0,state);
}

static constexpr unsigned float_decimal_length = 100000;
template<class T> __global__
static void randomize(curandState_t* state, T t, float lower_bound, float upper_bound) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < t.size(); i += blockDim.x * gridDim.x) {
		curandState_t tmpstate = *state;
		skipahead(i, &tmpstate);
		t[i] = curand(&tmpstate) % float_decimal_length;
		t[i] /= float_decimal_length;
		t[i] *= (upper_bound - lower_bound);
		t[i] += lower_bound;
	}

	__syncthreads();
	if (i == 0)
		skipahead(t.size(), state);
}
}

template<class SystemTag>
struct Random;

template<>
struct Random<device_tag> {

	static curandState_t* bc_curand_state() {
		struct bc_curandState_t {
		    curandState_t* state = nullptr;

			bc_curandState_t() {
				BC_CUDA_ASSERT((cudaMalloc((void**) &state, sizeof(curandState_t))));
				device_impl::bc_curand_init<<<1,1>>>(state);
				cudaDeviceSynchronize();
			}
			~bc_curandState_t() {
				BC_CUDA_ASSERT((cudaFree((void*)state)));
			}
		};
		thread_local bc_curandState_t bc_state;
		return bc_state.state;
	}

	template<class Stream, typename T>
	static void randomize(Stream stream, T t, float lower_bound, float upper_bound) {
		device_impl::randomize<<<calculate_block_dim(t.size()),calculate_threads(t.size()), 0, stream>>>(
				bc_curand_state(), t, lower_bound, upper_bound);
	}
};

}
}


#endif
#endif /* DEVICE_H_ */
