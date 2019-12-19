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

namespace bc {
namespace blas {

template<>
struct BLAS<device_tag> {

	template<class Stream>
	static void gemm(
			Stream stream,
			bool transA,
			bool transB,
			bc::size_t m,
			bc::size_t n,
			bc::size_t k,
			const float* alpha, const float* A, bc::size_t lda,
								const float* B, bc::size_t ldb,
			const float* beta,        float* C, bc::size_t ldc)
	{
		auto TRANS_A = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
		auto TRANS_B = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

		stream.enqueue([=]() {
			cublasHandle_t handle = stream.get_cublas_handle();
			BC_CUDA_ASSERT(
					(cublasSgemm(
							handle, TRANS_A, TRANS_B,
							m, n, k,
							alpha,
							A, lda,
							B, ldb,
							beta, C, ldc)));
		});
	}

	//y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
	template<class Stream>
	static void gemv(
			Stream stream,
			bool transA,
			bc::size_t m,
			bc::size_t n,
			const float* alpha, const float* A, bc::size_t lda,
								const float* X, bc::size_t incX,
			const float* beta,        float* Y, bc::size_t incY)
	{
		auto TRANS_A =  transA ? CUBLAS_OP_T : CUBLAS_OP_N;

		stream.enqueue([=]() {
			cublasHandle_t handle = stream.get_cublas_handle();
			BC_CUDA_ASSERT((cublasSgemv(handle, TRANS_A, m, n, alpha, A, lda, X, incX, beta, Y, incY)));
		});
	}

	template<class Stream>
	static void ger(Stream stream, int m, bc::size_t n,
			const float* alpha,
			const float* X, bc::size_t incX,
			const float* Y, bc::size_t incY,
				  float* A, bc::size_t lda)
	{
		stream.enqueue([=]() {
			cublasHandle_t handle = stream.get_cublas_handle();
			BC_CUDA_ASSERT((cublasSger(handle, m, n, alpha, X, incX, Y, incY, A, lda)));
		});
	}

	template<class Stream>
	static void dot(
			Stream stream,
			int n,
			float* A,
			const float* x, bc::size_t incX,
			const float* y, bc::size_t incY)
	{
		stream.enqueue([=]() {
			cublasHandle_t handle = stream.get_cublas_handle();
			BC_CUDA_ASSERT((cublasSdot(handle, n, x, incX, y, incY, A)));
		});
	}
};

}
}


#endif /* GPU_BLAS_H_ */
#endif
