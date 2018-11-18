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
#include <curand.h>
#include <curand_kernel.h>

namespace BC {
namespace gpu_impl {

template<int i> __device__  int tid() { return 0; }

//TODO SWITCH to recursive template iterator
//BUG  -nvcc struggles to compile std::enable_if_t in kernels
//template<> __device__ int tid<0>() { return blockIdx.x * blockDim.x + threadIdx.x; }
//template<> __device__ int tid<1>() { return blockIdx.y * blockDim.y + threadIdx.y; }
//template<> __device__ int tid<2>() { return blockIdx.z * blockDim.z + threadIdx.z; }
//
//template<int t> __device__ int tid_inc() { return 1; }
//template<> __device__ int tid_inc<0>() { return blockDim.x * gridDim.x; }
//template<> __device__ int tid_inc<1>() { return blockDim.y * gridDim.y; }
//template<> __device__ int tid_inc<2>() { return blockDim.z * gridDim.z; }
//
//
//
//template<int dim, int tid_x, class expression_t, class... indexes> __device__
//static std::enable_if_t<(dim == 0)> dev_evaluate(expression_t expr, indexes... ids) {
//    for (int i = tid<tid_x>(); i < expr.dimension(dim); i += tid_inc<tid_x>())
//        expr(i, ids...);
//}
//template<int dim, int tid_x, class expression_t, class... indexes> __device__
//static std::enable_if_t<(dim > 0)> dev_evaluate(expression_t expr, indexes... ids) {
//    for (int i = tid<tid_x>(); i < expr.dimension(dim); i += tid_inc<tid_x>())
//    return dev_evaluate<dim - 1, tid_x + 1>(expr, i, ids...);
//}
//
//    template<int dim, int tid_x, class expression_t, class... indexes> __global__
//    static std::enable_if_t<(dim > 0)> evaluate(expression_t expr, indexes... ids) {
//        for (int i = tid<tid_x>(); i < expr.dimension(dim); i += tid_inc<tid_x>())
//        return dev_evaluate<dim - 1, tid_x + 1>(expr, i, ids...);
//    }
//    template<int dim, int tid_x, class expression_t, class... indexes> __global__
//    static std::enable_if_t<(dim == 0)> evaluate(expression_t expr, indexes... ids) {
//        for (int i = tid<tid_x>(); i < expr.dimension(dim); i += tid_inc<tid_x>())
//            expr(i, ids...);
//    }

//};
//
template<class T> __global__
static void eval(T t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < t.size(); i += blockDim.x * gridDim.x) {
        t[i];
    }
}

template<typename T> __global__  static void eval2d(T t) {
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    for (; n < t.cols(); n += blockDim.y * gridDim.y) {

        int m = blockIdx.x * blockDim.x + threadIdx.x;
        for (; m < t.rows(); m += blockDim.x * gridDim.x) {
            t(m, n);
        }
    }
}
template<typename T> __global__ static void eval3d(T t) {
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    for (; k < t.dimension(2); k += blockDim.z * gridDim.z) {

    	int n = blockIdx.y * blockDim.y + threadIdx.y;
        for (; n < t.cols(); n += blockDim.y * gridDim.y) {

            int m = blockIdx.x * blockDim.x + threadIdx.x;
            for (; m < t.rows(); m += blockDim.x * gridDim.x) {
                t(m,n,k);
            }
        }
    }
}
//dont know how to do this
template<typename T> __global__ static void eval4d(T t) {
    int l = blockIdx.z * blockDim.z + threadIdx.z;
    for (; l < t.dimension(3); l += blockDim.z * gridDim.z) {

        int k = blockIdx.y * blockDim.y + threadIdx.y;
        for (;k < t.dimension(2); k += blockDim.y * gridDim.y) {

            int n = blockIdx.x * blockDim.x + threadIdx.x;
            for (; n < t.cols(); n += blockDim.x * gridDim.x) {

                for (int m = 0; m < t.rows(); ++m) {
                    t(m,n,k,l);
                }
            }
        }
    }
}
//don't know how to do this
template<typename T> __global__ static void eval5d(T t) {
    int p = blockIdx.z * blockDim.z + threadIdx.z;
    for (; p < t.dimension(4); p += blockDim.z * gridDim.z) {

        int l = blockIdx.y * blockDim.y + threadIdx.y;
        for (; l < t.dimension(3); l += blockDim.y * gridDim.y) {

            int k = blockIdx.x * blockDim.x + threadIdx.x;
            for (; k < t.dimension(2); k += blockDim.x * gridDim.x) {

                for (int n = 0; n < t.dimension(1); ++n) {

                    for (int m = 0; m < t.dimension(0); ++m) {
                        t(m, n, k, l, p);
                    }
                }
            }
        }
    }
}

template<class T, class U, class V> __global__
static void scalar_mul(T* t, U* u, V* v) {
    *t = *u * *v;
}
template<class T, class U, class V> __global__
static void scalar_mul(T* t, U u, V* v) {
    *t = u * *v;
}
template<class T, class U, class V> __global__
static void scalar_mul(T* t, U* u, V v) {
    *t = *u * v;
}
template<class T, class U, class V> __global__
static void scalar_mul(T* t, U u, V v) {
    *t = u * v;
}

template<class T> __global__
static void randomize(T t, float lower_bound, float upper_bound, int seed) {

     curandState_t state;
      curand_init(seed, /* the seed controls the sequence of random values that are produced */
                  seed, /* the sequence number is only important with multiple cores */
                  1, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                  &state);

    constexpr int floating_point_decimal_length = 10000;

    for (int i = 0; i < t.size(); ++i) {
        t[i] = curand(&state) % floating_point_decimal_length;
        t[i] /= floating_point_decimal_length;
        t[i] *= (upper_bound - lower_bound);
        t[i] += lower_bound;
    }
}

}


}

#endif
#endif //cudacc
