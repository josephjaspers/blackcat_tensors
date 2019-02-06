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
    template<class Context, typename T>
    static void randomize(Context context, T t, float lower_bound, float upper_bound) {
    	device_impl::randomize<<<blocks(t.size()),threads(), 0, context.get_stream()>>>(t, lower_bound, upper_bound, std::rand());
    }
};

}
}


#endif
#endif /* DEVICE_H_ */
