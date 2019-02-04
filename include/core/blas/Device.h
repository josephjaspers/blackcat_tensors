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

		template<class Context>
		static void scalar_mul(float* eval, float* a, float* b, Context context) {

			device_impl::scalar_mul<<<1, 1, 0, context.get_cuda_stream()>>>(eval, a, b);
		}

		template<class Context>
		static void scalar_mul(float* eval, float a, float* b, Context context) {

			device_impl::scalar_mul<<<1, 1, 0, context.get_cuda_stream()>>>(eval, a, b);
		}

		template<class Context>
		static void scalar_mul(float* eval, float* a, float b, Context context) {

			device_impl::scalar_mul<<<1, 1, 0, context.get_cuda_stream()>>>(eval, a, b);
		}

		template<class Context>
		static void scalar_mul(float* eval, float a, float b, Context context) {

			device_impl::scalar_mul<<<1, 1, 0, context.get_cuda_stream()>>>(eval, a, b);
		}

	template<class Context>
    static void gemm(Context context, bool transA, bool transB, BC::size_t  m, BC::size_t  n, BC::size_t  k,
            const float* alpha, const float* A, BC::size_t  lda,
                                const float* B, BC::size_t  ldb,
            const float* beta,           float* C, BC::size_t  ldc) {
        auto TRANS_A = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        auto TRANS_B = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasHandle_t& handle = context.get_cublas_handle();
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasSgemm(handle, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

	//y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
	template<class Context>
	static void gemv(Context context, bool transA, BC::size_t  m, BC::size_t  n,
			const float* alpha, const float* A, BC::size_t  lda,
								 const float* X, BC::size_t  incX,
			const float* beta,        float* Y, BC::size_t  incY) {


		auto TRANS_A =  transA ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasHandle_t& handle = context.get_cublas_handle();
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
		cublasSgemv(handle, TRANS_A, m, n, alpha, A, lda, X, incX, beta, Y, incY);
	}

	template<class Context>
	static void ger(Context context, int m, BC::size_t  n,
			const float* alpha,
								 const float* X, BC::size_t  incX,
								 const float* Y, BC::size_t  incY,
								  float* A, BC::size_t  lda) {

		cublasHandle_t& handle = context.get_cublas_handle();
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
		cublasSger(handle, m, n, alpha, X, incX, Y, incY, A, lda);
	}

	template<class Context>
	static void dot(Context context, int n, float* A, const float* x, BC::size_t  incX, const float* y, BC::size_t  incY) {
		cublasHandle_t& handle = context.get_cublas_handle();
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
		cublasSdot(handle, n, x, incX, y, incY, A);
	}
};

}
}


#endif /* GPU_BLAS_H_ */
#endif
