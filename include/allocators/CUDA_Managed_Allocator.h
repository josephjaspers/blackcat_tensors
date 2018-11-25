/*
 * CUda_Managed_Allocator.h
 *
 *  Created on: Oct 24, 2018
 *      Author: joseph
 */
#ifdef __CUDACC__
#ifndef CUDA_MANAGED_ALLOCATOR_H_
#define CUDA_MANAGED_ALLOCATOR_H_

#include "CUDA_Allocator.h"

namespace BC {
namespace module {
namespace stl {

struct CUDA_Managed_Allocator : CUDA_Allocator {

    template<typename T>
    static T*& allocate(T*& t, int sz=1) {
        cudaMallocManaged((void**) &t, sizeof(T) * sz);
        return t;
    }
};

}
}
}


#endif /* CUDA_MANAGED_ALLOCATOR_H_ */
#endif
