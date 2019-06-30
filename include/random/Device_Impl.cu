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


__global__
static void bc_curand_init(curandState_t* state) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i==0)
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
}
}

#endif
#endif /* DEVICE_IMPL_H_ */
