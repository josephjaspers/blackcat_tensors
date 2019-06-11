/*
 * Device.h
 *
 *  Created on: Dec 3, 2018
 *      Author: joseph
 */


#ifdef __CUDACC__
#ifndef DEVICE_H_
#define DEVICE_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "Device_Impl.cu"

namespace BC {
namespace random {

template<>
struct Random<device_tag> {

	static curandState_t* bc_curand_state() {
		struct bc_curandState_t {
		    curandState_t* state = nullptr;

			bc_curandState_t() {
				BC_CUDA_ASSERT((cudaMalloc((void**) &state, sizeof(curandState_t))));
			}
			~bc_curandState_t() {
		    	BC_CUDA_ASSERT((cudaFree((void*)state)));
			}
		};
		bc_curandState_t bc_state;
		return bc_state.state;
	}

    template<class Stream, typename T>
    static void randomize(Stream stream, T t, float lower_bound, float upper_bound) {
    	device_impl::randomize<<<blocks(t.size()),threads(), 0, stream>>>(
    			bc_curand_state(), t, lower_bound, upper_bound, std::rand());
    }
};

}
}


#endif
#endif /* DEVICE_H_ */
