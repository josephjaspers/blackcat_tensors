/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef GPU_CONSTANTS_H_
#define GPU_CONSTANTS_H_

namespace BC {

template<class core_lib>
struct GPU_Constants {

    static float* static_allocate(float value) {
        float* t;
        cudaMallocManaged((void**) &t, sizeof(float));
        cudaMemcpy(t, &value, sizeof(float), cudaMemcpyHostToDevice);
        return t;
    }

    static void scalar_mul(float* eval, float* a, float* b) {

        gpu_impl::scalar_mul<<<1, 1>>>(eval, a, b);
        cudaDeviceSynchronize();
    }
    static void scalar_mul(float* eval, float a, float* b) {

        gpu_impl::scalar_mul<<<1, 1>>>(eval, a, b);
        cudaDeviceSynchronize();
    }
    static void scalar_mul(float* eval, float* a, float b) {

        gpu_impl::scalar_mul<<<1, 1>>>(eval, a, b);
        cudaDeviceSynchronize();
    }
    static void scalar_mul(float* eval, float a, float b) {

        gpu_impl::scalar_mul<<<1, 1>>>(eval, a, b);
        cudaDeviceSynchronize();
    }

};

}



#endif /* GPU_CONSTANTS_H_ */
