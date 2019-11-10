/*
 * host.h
 *
 *  Created on: Dec 3, 2018
 *	  Author: joseph
 */

#ifndef BC_BLAS_HOST_H_
#define BC_BLAS_HOST_H_

#ifdef BC_BLAS_USE_C_LINKAGE
#define BC_EXTERN_C_BEGIN extern "C" {
#define BC_EXTERN_C_END }
#else
#define BC_EXTERN_C_BEGIN
#define BC_EXTERN_C_END	
#endif 

#if __has_include(<cblas.h>)

BC_EXTERN_C_BEGIN
#include <cblas.h>
BC_EXTERN_C_END

#elif __has_include(<mkl.h>)
#include <mkl.h>
#else
#ifndef _MSC_VER
#warning "BLACKCAT_TENSORS REQUIRES A VALID <cblas.h> OR <mkl.h> IN ITS PATH"
#endif
#endif

#undef BC_EXTERN_C_BEGIN
#undef BC_EXTERN_C_END 

namespace BC {
namespace blas {
/*
 * creates a BLAS wrapper for BC_Tensors
 * -> uses generic function names but without the prefix of s/d for precision type.
 *  The automatic template deduction chooses the correct path
 *
 *  (this is to enable BC_Tensors's to not have to specialize the template system)
 */
template<>
struct BLAS<host_tag> {

	/*
	 * a = M x K
	 * b = K x N
	 * c = M x N
	 */

	template<class Stream>
	static void gemm(
			Stream stream, bool transA, bool transB,
			BC::size_t m, BC::size_t n, BC::size_t k,
			const float* alpha, const float* A, BC::size_t lda,
			                    const float* B, BC::size_t ldb,
			const float* beta,        float* C, BC::size_t ldc)
	{
		auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;
		auto TRANS_B =  transB ? CblasTrans : CblasNoTrans;

		stream.enqueue([=]() {
				cblas_sgemm(
						CblasColMajor,
						TRANS_A, TRANS_B, m, n, k,
						*alpha, A, lda, B, ldb, *beta, C, ldc);
		});
	}

	template<class Stream>
	static void gemm(
			Stream stream, bool transA, bool transB,
			BC::size_t m, BC::size_t n, BC::size_t k,
			const double *alpha, const double* A, BC::size_t lda,
			                     const double* B, BC::size_t ldb,
			const double *beta,        double* C, BC::size_t ldc) {

		auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;
		auto TRANS_B =  transB ? CblasTrans : CblasNoTrans;

		stream.enqueue([=]() {
			cblas_dgemm(
					CblasColMajor,
					TRANS_A, TRANS_B, m, n, k,
					*alpha, A, lda, B, ldb, *beta, C, ldc);
		});

	}

	//y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
	template<class Stream>
	static void gemv(
			Stream stream, bool transA,BC::size_t m, BC::size_t n,
			const double* alpha, const double* A, BC::size_t lda,
			                     const double* X, BC::size_t incX,
			const double* beta,        double* Y, BC::size_t incY) {

		auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;

		stream.enqueue([=]() {
			cblas_dgemv(
					CblasColMajor,
					TRANS_A, m, n,
					*alpha, A, lda, X, incX, *beta, Y, incY);
		});
	}

	template<class Stream>
	static void gemv(
			Stream stream, bool transA, BC::size_t m, BC::size_t n,
			const float* alpha, const float* A, BC::size_t lda,
			                    const float* X, BC::size_t incX,
			const float* beta,        float* Y, BC::size_t incY)
	{
		auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;

		stream.enqueue([=]() {
			cblas_sgemv(
					CblasColMajor,
					TRANS_A, m, n,
					*alpha, A, lda, X, incX, *beta, Y, incY);
		});
	}

	template<class Stream>
	static void ger(Stream stream, int m, BC::size_t n,
			const double* alpha,
			const double* X, BC::size_t incX,
			const double* Y, BC::size_t incY,
			      double* A, BC::size_t lda) {

		stream.enqueue([=]() {
			cblas_dger(
					CblasColMajor,
					m, n,
					*alpha, X, incX, Y, incY, A, lda);
		});
	}

	template<class Stream>
	static void ger(Stream stream, int m, BC::size_t n,
			const float* alpha,
			const float* X, BC::size_t incX,
			const float* Y, BC::size_t incY,
			float* A, BC::size_t lda) {

		stream.enqueue([=]() {
			cblas_sger(
					CblasColMajor,
					m, n,
					*alpha, X, incX, Y, incY, A, lda);
		});
	}

	template<class Stream>
	static void dot(Stream stream, int n,
			double* A,
			const double* x, BC::size_t incX,
			const double* y, BC::size_t incY) {

		stream.enqueue([=]() {
			*A = cblas_ddot(n, x, incX, y, incY);
		});
	}

	template<class Stream>
	static void dot(Stream stream, int n,
			float* A,
			const float* x, BC::size_t incX,
			const float* y, BC::size_t incY) {

		stream.enqueue([=]() {
			*A = cblas_sdot(n, x, incX, y, incY);
		});
	}
};
}

}


#endif /* HOST_H_ */
