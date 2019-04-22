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

struct Device {

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

    template<class Context, typename T>
    static void randomize(Context context, T t, float lower_bound, float upper_bound) {
    	device_impl::randomize<<<blocks(t.size()),threads(), 0, context.get_stream()>>>(
    			bc_curand_state(), t, lower_bound, upper_bound, std::rand());
    }
};

}
}


#endif
#endif /* DEVICE_H_ */
