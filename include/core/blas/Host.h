/*
 * host.h
 *
 *  Created on: Dec 3, 2018
 *      Author: joseph
 */

#ifndef BC_BLAS_HOST_H_
#define BC_BLAS_HOST_H_

#include <cblas.h>
//#include <mkl_cblas.h> //TODO create/ifdef wrapper for MKL
#include "Host_Impl.h"

namespace BC {
namespace blas {
/*
 * creates a BLAS wrapper for BC_Tensors
 * -> uses generic function names but without the prefix of s/d for precision type.
 *  The automatic template deduction chooses the correct path
 *
 *  (this is to enable BC_Tensors's to not have to specialize the template system)
 */

struct Host {

    /*
     * a = M x K
     * b = K x N
     * c = M x N
     */

	template<class Context, class OutputScalar, class... Scalars>
	static void calculate_alpha(Context, OutputScalar& eval, Scalars... scalars) {
		eval = host_impl::calculate_alpha(scalars...);
	}
	template<class Context, class OutputScalar, class... Scalars>
	static void calculate_alpha(Context, OutputScalar* eval, Scalars... scalars) {
		eval[0] = host_impl::calculate_alpha(scalars...);
	}

	template<class value_type, int value>
	static const value_type& scalar_constant() {
		static value_type val = value;
		return val;
	}

	template<class Context>
    static void gemm(Context context, bool transA, bool transB, BC::size_t  m, BC::size_t  n, BC::size_t  k,
            const float alpha, const float* A, BC::size_t  lda,
                                const float* B, BC::size_t  ldb,
            const float beta,           float* C, BC::size_t  ldc) {

        auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;
        auto TRANS_B =  transB ? CblasTrans : CblasNoTrans;

        context.get_stream().push_job([=]() {
                cblas_sgemm(CblasColMajor, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		});
    }

	template<class Context>
    static void gemm(Context context, bool transA, bool transB, BC::size_t  m, BC::size_t  n, BC::size_t  k,
            const double alpha, const double* A, BC::size_t  lda,
                                 const double* B, BC::size_t  ldb,
            const double beta,        double* C, BC::size_t  ldc) {

        auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;
        auto TRANS_B =  transB ? CblasTrans : CblasNoTrans;

        context.push_job([=]() {
        	cblas_dgemm(CblasColMajor, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    	});

	}

    //y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
	template<class Context>
	static void gemv(Context context, bool transA, BC::size_t  m, BC::size_t  n,
            const double alpha, const double* A, BC::size_t  lda,
                                 const double* X, BC::size_t  incX,
            const double beta,        double* Y, BC::size_t  incY) {

        auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;

        context.push_job([=]() {
        	cblas_dgemv(CblasColMajor, TRANS_A, m, n, alpha, A, lda, X, incX, beta, Y, incY);
    	});
	}

	template<class Context>
    static void gemv(Context context, bool transA, BC::size_t  m, BC::size_t  n,
            const float alpha, const float* A, BC::size_t  lda,
                                const float* X, BC::size_t  incX,
            const float beta,              float* Y, BC::size_t  incY) {

        auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;

        context.push_job([=]() {
        	cblas_sgemv(CblasColMajor, TRANS_A, m, n, alpha, A, lda, X, incX, beta, Y, incY);
    	});
	}

	template<class Context>
    static void ger(Context context, int m, BC::size_t  n,
            const double alpha,
                                 const double* X, BC::size_t  incX,
                                 const double* Y, BC::size_t  incY,
                                  double* A, BC::size_t  lda) {

        context.push_job([=]() {
        	cblas_dger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
        });
	}

	template<class Context>
    static void ger(Context context, int m, BC::size_t  n,
            const float alpha,
                                 const float* X, BC::size_t  incX,
                                 const float* Y, BC::size_t  incY,
                                  float* A, BC::size_t  lda) {

        context.push_job([=]() {
        	cblas_sger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
        });
	}

	template<class Context>
    static void dot(Context context, int n, double* A, const double* x, BC::size_t  incX, const double* y, BC::size_t  incY) {
        context.push_job([=]() {
        	*A = cblas_ddot(n, x, incX, y, incY);
        });
    }

	template<class Context>
    static void dot(Context context, int n, float* A, const float* x, BC::size_t  incX, const float* y, BC::size_t  incY) {
        context.push_job([=]() {
        	*A = cblas_sdot(n, x, incX, y, incY);
        });
	}
};
}

}


#endif /* HOST_H_ */
