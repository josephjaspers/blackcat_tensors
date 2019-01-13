/*
 * CUda_Managed_Allocator.h
 *
 *  Created on: Oct 24, 2018
 *      Author: joseph
 */
#ifdef __CUDACC__
#ifndef CUDA_MANAGED_ALLOCATOR_H_
#define CUDA_MANAGED_ALLOCATOR_H_

#include "Device.h"

namespace BC {
namespace allocator {

template<class T>
struct Device_Managed : Device<T> {

	template<class scalar_t>
    struct rebind {
    	using other = Device_Managed<scalar_t>;
    };

    T* allocate(BC::size_t sz) {
    	T* memptr = nullptr;
        cudaMallocManaged((void**) &memptr, sizeof(T) * sz);
        return memptr;
    }
};

}
}


#endif /* CUDA_MANAGED_ALLOCATOR_H_ */
#endif
