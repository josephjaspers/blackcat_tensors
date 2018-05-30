/*
 * Mathematics_CPU_BLAS.h
 *
 *  Created on: May 6, 2018
 *      Author: joseph
 */

#ifndef MATHEMATICS_CPU_BLAS_H_
#define MATHEMATICS_CPU_BLAS_H_
#include "cblas.h"

namespace BC {

/*
 * creates a BLAS wrapper for BC_Tensors
 * -> uses generic function names but without the prefix of s/d for precision type.
 *  The automatic template deduction chooses the correct path
 *
 *  (this is to enable BC_Tensors's to not have to specialize the template system)
 */

template<class core_lib>
struct CPU_BLAS  {

	template<class U, class T, class V>
	static void scalarMul(U eval, T a, V b) {
		*eval = a[0] * b[0];
	}

	/*
	 * a = M x K
	 * b = K x N
	 * c = M x N
	 */

	static void vec_copy(int n, float* x, int incx, float* y, int incy) { return cblas_scopy(n, x, incx, y, incy); }
	static void vec_copy(int n, double* x, int incx, double* y, int incy) { return cblas_dcopy(n, x, incx, y, incy); }

	static void gemm(bool transA, bool transB, const float* A, const float* B, float* C, int m, int n, int k,
			const float* scalarA = nullptr, const float* scalarB = nullptr,  int lda = 0, int ldb =0, int ldc =0) {

		auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;
		auto TRANS_B =  transB ? CblasTrans : CblasNoTrans;

		if (lda == 0 ) lda = m;
		if (ldb == 0 ) ldb = n;
		if (ldc == 0 ) ldc = m;

	    const float beta   =  scalarB ? *scalarB : 0;
	    const float alpha  =  scalarA ? *scalarA : 1;

		cblas_sgemm(CblasColMajor, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	static void gemm(bool transA, bool transB, const double* A, const double* B, double* C, int m, int n, int k,
			const double* scalarA = nullptr, const double* scalarB = nullptr, int lda = 0, int ldb = 0, int ldc = 0) {

		auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;
		auto TRANS_B =  transB ? CblasTrans : CblasNoTrans;

		if (lda == 0 ) lda = m;
		if (ldb == 0 ) ldb = n;
		if (ldc == 0 ) ldc = m;

		const double beta   =  scalarB ? *scalarB : 0.0;
		const double alpha  =  scalarA ? *scalarA : 1.0;

		cblas_dgemm(CblasColMajor, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
};
}

#endif /* MATHEMATICS_CPU_BLAS_H_ */
