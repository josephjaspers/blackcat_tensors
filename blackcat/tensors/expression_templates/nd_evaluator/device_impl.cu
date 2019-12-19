/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef __CUDACC__
#ifndef BC_GPU_IMPL
#define BC_GPU_IMPL

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace BC {
namespace tensors {
namespace exprs {
namespace evaluator {
namespace gpu_impl {

template<class T> __global__
static void eval(T t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < t.size(); i += blockDim.x * gridDim.x) {
        t[i];
    }
}

template<typename T> __global__  static void eval2d(T t) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    for (; n < t.cols(); n += blockDim.x * gridDim.x) {

        int m = blockIdx.y * blockDim.y + threadIdx.y;
        for (; m < t.rows(); m += blockDim.y * gridDim.y) {
            t(n, m);
        }
    }
}
template<typename T> __global__ static void eval3d(T t) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    for (; k < t.dimension(2); k += blockDim.x * gridDim.x) {

    	int n = blockIdx.y * blockDim.y + threadIdx.y;
        for (; n < t.cols(); n += blockDim.y * gridDim.y) {

            int m = blockIdx.z * blockDim.z + threadIdx.z;
            for (; m < t.rows(); m += blockDim.z * gridDim.z) {
                t(k,n,m);
            }
        }
    }
}
//dont know how to do this
template<typename T> __global__ static void eval4d(T t) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    for (; l < t.dimension(3); l += blockDim.x * gridDim.x) {

        int k = blockIdx.y * blockDim.y + threadIdx.y;
        for (;k < t.dimension(2); k += blockDim.y * gridDim.y) {

            int n = blockIdx.z * blockDim.z + threadIdx.z;
            for (; n < t.cols(); n += blockDim.z * gridDim.z) {

                for (int m = 0; m < t.rows(); ++m) {
                    t(l, k, n, m);
                }
            }
        }
    }
}
//don't know how to do this
template<typename T> __global__ static void eval5d(T t) {
    int p = blockIdx.z * blockDim.x + threadIdx.x;
    for (; p < t.dimension(4); p += blockDim.x * gridDim.x) {

        int l = blockIdx.y * blockDim.y + threadIdx.y;
        for (; l < t.dimension(3); l += blockDim.y * gridDim.y) {

            int k = blockIdx.x * blockDim.z + threadIdx.z;
            for (; k < t.dimension(2); k += blockDim.z * gridDim.z) {

                for (int n = 0; n < t.dimension(1); ++n) {

                    for (int m = 0; m < t.dimension(0); ++m) {
                        t(p, l, k, n, m);
                    }
                }
            }
        }
    }
}

}
}
}
}
}

#endif
#endif //cudacc
