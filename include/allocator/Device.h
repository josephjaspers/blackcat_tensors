
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

struct Device {

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

};

}
}




#endif /* CUDA_ALLOCATOR_H_ */
#endif
