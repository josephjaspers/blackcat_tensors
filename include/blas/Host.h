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

	template<class T, class enabler = void>
	struct get_value_impl {
	    static auto impl(T scalar) {
	        return scalar;
	    }
	};
	template<class T>
	struct get_value_impl<T, std::enable_if_t<!std::is_same<decltype(std::declval<T&>()[0]), void>::value>>  {
	    static auto impl(T scalar) {
	        return scalar[0];
	    }
	};

	template<class T>
	static auto get_value(T scalar) {
		return get_value_impl<T>::impl(scalar);
	}

	template<class U, class T, class V>
	static void scalar_mul(U& eval, const T& a, const V& b) {
		eval = get_value(a) * get_value(b);
	}
	template<class U, class T, class V>
	static void scalar_mul(U* eval, const T& a, const V& b) {
		eval[0] = get_value(a) * get_value(b);
	}


	template<class T>
	static T static_allocate(T value) {
		return T(value);
	}

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


#endif /* HOST_H_ */
