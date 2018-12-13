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

    T*& allocate(T*& t, int sz=1) {
        cudaMallocManaged((void**) &t, sizeof(T) * sz);
        return t;
    }
};

}
}


#endif /* CUDA_MANAGED_ALLOCATOR_H_ */
#endif
