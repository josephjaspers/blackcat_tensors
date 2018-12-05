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

	static float* static_allocate(float value) {
			float* t;
			cudaMallocManaged((void**) &t, sizeof(float));
			cudaMemcpy(t, &value, sizeof(float), cudaMemcpyHostToDevice);
			return t;
		}

		static void scalar_mul(float* eval, float* a, float* b) {

			device_impl::scalar_mul<<<1, 1>>>(eval, a, b);
			cudaDeviceSynchronize();
		}
		static void scalar_mul(float* eval, float a, float* b) {

			device_impl::scalar_mul<<<1, 1>>>(eval, a, b);
			cudaDeviceSynchronize();
		}
		static void scalar_mul(float* eval, float* a, float b) {

			device_impl::scalar_mul<<<1, 1>>>(eval, a, b);
			cudaDeviceSynchronize();
		}
		static void scalar_mul(float* eval, float a, float b) {

			device_impl::scalar_mul<<<1, 1>>>(eval, a, b);
			cudaDeviceSynchronize();
		}

    static void gemm(bool transA, bool transB, int m, int n, int k,
            const float* alpha, const float* A, int lda,
                                const float* B, int ldb,
            const float* beta,           float* C, int ldc) {
        auto TRANS_A = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        auto TRANS_B = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasSgemm(handle, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        cudaDeviceSynchronize();
        cublasDestroy(handle);
    }

	//y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
	static void gemv(bool transA, int m, int n,
			const float* alpha, const float* A, int lda,
								 const float* X, int incX,
			const float* beta,        float* Y, int incY) {


		auto TRANS_A =  transA ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
		cublasSgemv(handle, TRANS_A, m, n, alpha, A, lda, X, incX, beta, Y, incY);
		cudaDeviceSynchronize();
		cublasDestroy(handle);
	}

	static void ger(int m, int n,
			const float* alpha,
								 const float* X, int incX,
								 const float* Y, int incY,
								  float* A, int lda) {

		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
		cublasSger(handle, m, n, alpha, X, incX, Y, incY, A, lda);
		cudaDeviceSynchronize();
		cublasDestroy(handle);
	}

	static void dot(int n, float* A, const float* x, int incX, const float* y, int incY) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
		cublasSdot(handle, n, x, incX, y, incY, A);
		cudaDeviceSynchronize();
		cublasDestroy(handle);
	}
};

}
}


#endif /* GPU_BLAS_H_ */
#endif
