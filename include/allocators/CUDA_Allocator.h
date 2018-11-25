
/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifdef __CUDACC__
#ifndef CUDA_ALLOCATOR_H_
#define CUDA_ALLOCATOR_H_



namespace BC {

class device_tag;

namespace module {
namespace stl {

struct CUDA_Allocator : evaluator::Device {

    using mathlib_t = evaluator::Device;
    using system_tag = device_tag;

    template<typename T>
    static T*& allocate(T*& t, int sz=1) {
        cudaMalloc((void**) &t, sizeof(T) * sz);
        return t;
    }

    template<typename T>
    static void deallocate(T* t) {
        cudaFree((void*)t);
    }
    template<typename T>
    static void deallocate(T t) {
        //empty
    }

    template<class T>
    static void HostToDevice(T* t, const T* u, int size = 1) {
        cudaDeviceSynchronize();
        cudaMemcpy(t, u, sizeof(T) * size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    template<class T>
    static void DeviceToHost(T* t, const T* u, int size = 1) {
        cudaDeviceSynchronize();
        cudaMemcpy(t, u, sizeof(T) * size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    template<class T>
    static T extract(const T* data_ptr, int index) {
    	T host_data;
    	cudaDeviceSynchronize();
        cudaMemcpy(&host_data, &data_ptr[index], sizeof(T), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        return host_data;
    }
};

}
}
}



#endif /* CUDA_ALLOCATOR_H_ */
#endif
