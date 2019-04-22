/*
 * Device_Impl.h
 *
 *  Created on: Dec 3, 2018
 *      Author: joseph
 */

#ifdef __CUDACC__
#ifndef BC_RANDOM_DEVICE_DEVICE_IMPL_H_
#define BC_RANDOM_DEVICE_DEVICE_IMPL_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

namespace BC {
namespace random {
namespace device_impl {

template<class T> __global__
static void curand_initializer(curandState_t* state, int seed=1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
	    curand_init(seed, /* the seed controls the sequence of random values that are produced */
	                  seed, /* the sequence number is only important with multiple cores */
	                  1, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	                  state);
    }
}

static constexpr unsigned float_decimal_length = 100000;
template<class T> __global__
static void randomize(curandState_t* state, T t, float lower_bound, float upper_bound, int seed) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < t.size(); i += blockDim.x * gridDim.x) {
    	skipahead(i, state);
        t[i] = curand(state) % float_decimal_length;
        t[i] /= float_decimal_length;
        t[i] *= (upper_bound - lower_bound);
        t[i] += lower_bound;
    }

}
}
}
}

#endif
#endif /* DEVICE_IMPL_H_ */
