/*
 * Mathematics_CPU_BLAS.h
 *
 *  Created on: May 6, 2018
 *      Author: joseph
 */

#ifndef MATHEMATICS_CPU_BLAS_H_
#define MATHEMATICS_CPU_BLAS_H_
#include <cblas.h>
//#include <mkl_cblas.h> //TODO create/ifdef wrapper for MKL

namespace BC {
namespace evaluator {
namespace host_impl {
/*
 * creates a BLAS wrapper for BC_Tensors
 * -> uses generic function names but without the prefix of s/d for precision type.
 *  The automatic template deduction chooses the correct path
 *
 *  (this is to enable BC_Tensors's to not have to specialize the template system)
 */

template<class core_lib>
struct BLAS  {

    /*
     * a = M x K
     * b = K x N
     * c = M x N
     */

    static void gemm(bool transA, bool transB, int m, int n, int k,
            const float alpha, const float* A, int lda,
                                const float* B, int ldb,
            const float beta,           float* C, int ldc) {

        auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;
        auto TRANS_B =  transB ? CblasTrans : CblasNoTrans;

        cblas_sgemm(CblasColMajor, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    static void gemm(bool transA, bool transB, int m, int n, int k,
            const double alpha, const double* A, int lda,
                                 const double* B, int ldb,
            const double beta,        double* C, int ldc) {

        auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;
        auto TRANS_B =  transB ? CblasTrans : CblasNoTrans;

        cblas_dgemm(CblasColMajor, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    //y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
    static void gemv(bool transA, int m, int n,
            const double alpha, const double* A, int lda,
                                 const double* X, int incX,
            const double beta,        double* Y, int incY) {

        auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;

        cblas_dgemv(CblasColMajor, TRANS_A, m, n, alpha, A, lda, X, incX, beta, Y, incY);
    }
    static void gemv(bool transA, int m, int n,
            const float alpha, const float* A, int lda,
                                const float* X, int incX,
            const float beta,              float* Y, int incY) {

        auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;

        cblas_sgemv(CblasColMajor, TRANS_A, m, n, alpha, A, lda, X, incX, beta, Y, incY);
    }

    static void ger(int m, int n,
            const double alpha,
                                 const double* X, int incX,
                                 const double* Y, int incY,
                                  double* A, int lda) {

        cblas_dger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
    }
    static void ger(int m, int n,
            const float alpha,
                                 const float* X, int incX,
                                 const float* Y, int incY,
                                  float* A, int lda) {

        cblas_sger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
    }


    static void dot(int n, double* A, const double* x, int incX, const double* y, int incY) {
        *A = cblas_ddot(n, x, incX, y, incY);
    }
    static void dot(int n, float* A, const float* x, int incX, const float* y, int incY) {
        *A = cblas_sdot(n, x, incX, y, incY);
    }
};
}
}
}

#endif /* MATHEMATICS_CPU_BLAS_H_ */
