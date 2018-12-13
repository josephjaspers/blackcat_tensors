
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

namespace allocator {

template<class T, class derived=void>
struct Device : AllocatorBase<std::conditional_t<std::is_void<derived>::value, Device<T>, derived>> {

    using system_tag = device_tag;

    T* allocate(int sz=1) const {
    	T* data_ptr;
        cudaMalloc((void**) &data_ptr, sizeof(T) * sz);
        return data_ptr;
    }

    void deallocate(T* data_ptr) const {
        cudaFree((void*)data_ptr);
    }

};

}
}




#endif /* CUDA_ALLOCATOR_H_ */
#endif
