/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef __CUDACC__
#ifndef BC_BLAS_DEVICE_H_
#define BC_BLAS_DEVICE_H_

#include "Device_Impl.cu"

namespace BC {
namespace blas {

struct Device {


	template<class value_type, int x>
	static const value_type* beta_constant() {
	static_assert(x >= -1 || x <= 1, "INVALID ARGUMENT FOR BETA CONSTANT");

	static const value_type* zero = static_allocate<value_type>(0);
	static const value_type* one = static_allocate<value_type>(1);
	static const value_type* neg_one = static_allocate<value_type>(-1);
	cudaDeviceSynchronize();


	if (x == 0) {
		return zero;
	} else if (x == 1) {
		return one;
	} else if (x == -1) {
		return neg_one;
	} else {
		return nullptr;
	}
}

		template<class scalar_type>
		static scalar_type* static_allocate(scalar_type value) {
			scalar_type* ret_val;
			cudaMallocManaged((void**) &ret_val, sizeof(scalar_type));
			cudaMemcpy(ret_val, &value, sizeof(scalar_type), cudaMemcpyHostToDevice);
			return ret_val;
		}


		static void scalar_mul(float* eval, float* a, float* b) {

			device_impl::scalar_mul<<<1, 1>>>(eval, a, b);
		}
		static void scalar_mul(float* eval, float a, float* b) {

			device_impl::scalar_mul<<<1, 1>>>(eval, a, b);
		}
		static void scalar_mul(float* eval, float* a, float b) {

			device_impl::scalar_mul<<<1, 1>>>(eval, a, b);
		}
		static void scalar_mul(float* eval, float a, float b) {

			device_impl::scalar_mul<<<1, 1>>>(eval, a, b);
		}

	template<bool HostPointer>
    static void gemm(bool transA, bool transB, BC::size_t  m, BC::size_t  n, BC::size_t  k,
            const float* alpha, const float* A, BC::size_t  lda,
                                const float* B, BC::size_t  ldb,
            const float* beta,           float* C, BC::size_t  ldc) {
        auto TRANS_A = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        auto TRANS_B = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        cublasHandle_t handle;
        cublasCreate(&handle);
		cublasSetPointerMode(handle, HostPointer? CUBLAS_POINTER_MODE_HOST : CUBLAS_POINTER_MODE_DEVICE);
        cublasSgemm(handle, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        cublasDestroy(handle);
    }

	//y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
	template<bool HostPointer>
    static void gemv(bool transA, BC::size_t  m, BC::size_t  n,
			const float* alpha, const float* A, BC::size_t  lda,
								 const float* X, BC::size_t  incX,
			const float* beta,        float* Y, BC::size_t  incY) {


		auto TRANS_A =  transA ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSetPointerMode(handle, HostPointer? CUBLAS_POINTER_MODE_HOST : CUBLAS_POINTER_MODE_DEVICE);
		cublasSgemv(handle, TRANS_A, m, n, alpha, A, lda, X, incX, beta, Y, incY);
		cublasDestroy(handle);
	}

	template<bool HostPointer>
	static void ger(int m, BC::size_t  n,
			const float* alpha,
								 const float* X, BC::size_t  incX,
								 const float* Y, BC::size_t  incY,
								  float* A, BC::size_t  lda) {

		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSetPointerMode(handle, HostPointer? CUBLAS_POINTER_MODE_HOST : CUBLAS_POINTER_MODE_DEVICE);
		cublasSger(handle, m, n, alpha, X, incX, Y, incY, A, lda);
		cublasDestroy(handle);
	}

	template<bool HostPointer>
	static void dot(int n, float* A, const float* x, BC::size_t  incX, const float* y, BC::size_t  incY) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSetPointerMode(handle, HostPointer? CUBLAS_POINTER_MODE_HOST : CUBLAS_POINTER_MODE_DEVICE);
		cublasSdot(handle, n, x, incX, y, incY, A);
		cublasDestroy(handle);
	}
};

}
}


#endif /* GPU_BLAS_H_ */
#endif
