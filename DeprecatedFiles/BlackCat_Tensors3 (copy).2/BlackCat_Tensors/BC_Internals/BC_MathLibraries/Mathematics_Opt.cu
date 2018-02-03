/*
 * Mathematics_Opt.cu
 *
 *  Created on: Jan 12, 2018
 *      Author: joseph
 */

#ifndef MATHEMATICS_OPT_CU_
#define MATHEMATICS_OPT_CU_

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "BC_PrintFunctions.h"
#include "Mathematics_CPU.h"

namespace BC {
class OPT {
public:
	/*
	 * T -- An object (Generally an array) with a functional [] operator
	 * J -- Either a value on the stack to be assigned or another array similar to above.
	 *
	 * J -- may also be a single value passed by pointer, certains methods are overloaded to handle these instance
	 *
	 */

	//destructor - no destructor are for controlling destruction of the pointer
	template<typename T>
	static void initialize(T*& t, int sz) {
		t = new T[sz];
	}
	template<typename T>
	static void unified_initialize(T*& t, int sz) {
		t = new T[sz];
	}
	template<typename T>
	static void destroy(T* t) {
		delete[] t;
	}
	template<typename T, typename J>
	static void fill(T& t, const J j, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = j;
		}
#pragma omp barrier
	}

	template<typename T>
	static void zeros(T& t, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = 0;
		}
#pragma omp barrier
	}

	template<typename T, typename J>
	static void fill(T& t, const J* j, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = *j;
		}
#pragma omp barrier
	}

	template<typename T>
	static void eval(T&& t, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i];
		}
#pragma omp barrier
	}
	template<typename T, typename J>
	static void set_heap(T *t, J *j) {
		*t = *j;
	}
	template<typename T, typename J>
	static void set_stack(T *t, J j) {
		*t = j;
	}

	template<typename T>
	static void zero(T t, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = 0;
		}
#pragma omp barrier
	}
	template<typename T, typename J>
	static void copy(T t, J j, int sz) {
		if (sz < BC_CPU_SINGLE_THREAD_THRESHOLD) {
			copy_single_thread(t, j, sz);
			return;
		}
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = j[i];
		}
#pragma omp barrier
	}

	template<typename T, typename J>
	static void copy_single_thread(T& t, const J& j, int sz) {
		for (int i = 0; i < sz; ++i) {
			t[i] = j[i];
		}
	}
	template<typename T, typename J>
	static void copy_stack(T t, J j, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = j[i];
		}
#pragma omp barrier
	}
	template<typename T, typename J>
	static void randomize(T t, J lower_bound, J upper_bound, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = ((double) (rand() / ((double) RAND_MAX + 1)) * (upper_bound - lower_bound)) + lower_bound;

		}
#pragma omp barrier
	}
	template<class T, class RANKS>
	static void print(const T ary, const RANKS ranks, int order, int print_length) {
		BC::print(ary, ranks, order, print_length);
	}

	/*
	 * a = M x N
	 * b = N x K
	 * c = M x K
	 */

	//This doesn't actually give a shit if transposed --- just using this till I integrate blas
	template<class A, class B, class C>
	static void MatrixMul(bool transA, bool transB, A a, B b,  C c, int m, int n, int k, int lda = 0, int ldb = 0, int ldc = 0) {
		if (lda == 0 and ldb == 0 and ldc ==0) {
			lda = m;
			ldb = n;
			ldc = m;
		}
#pragma omp parallel for
		for (int z = 0; z < k; ++z) {
			for (int x = 0; x < n; ++x) {
				for (int y = 0; y < m; ++y) {
					c[z * ldc + y] += a[x * lda + y] * b[z * ldb + x];

				}
			}
		}
#pragma omp barrier
	}
	/*
	 * a = M x N
	 * b = N x K
	 * c = M x K
	 */

	  template<class T>
	  static double* aryToFloat(T ary, int sz) {
		  float* conv = new float[sz];
#pragma omp parallel for
		  for (int i = 0; i < sz; ++i) {
			  conv[i] = ary[i];
		  }
#pragma omp barrier

	  }
//	  template<class T>
//	  static void MatrixMul(bool transA, bool transB, const T *A, const T *B, T *C,
//			  const int m, const int k, const int n, int lda = 0, int ldb = 0, int ldc = 0) {
//
//
//	}
	  static void MatrixMul(bool transA, bool transB, const float *A, const float *B, float *C,
			  const int m, const int k, const int n, int lda = 0, int ldb = 0, int ldc = 0) {

		  if (lda == 0 ) lda = m;
		  if (ldb == 0 ) ldb = n;
		  if (ldc == 0 ) ldc = m;

	     const float alf = 1;
	     const float bet = 0;
	     const float *alpha = &alf;
	     const float *beta = &bet;

	     // Create a handle for CUBLAS
	     cublasHandle_t handle;
	     cublasCreate(&handle);

	     auto TRANS_A =  transA ? CUBLAS_OP_T : CUBLAS_OP_N;
	     auto TRANS_B =  transA ? CUBLAS_OP_T : CUBLAS_OP_N;

	     // Do the actual multiplication
	     cublasSgemm(handle, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	     cudaDeviceSynchronize();
	     cublasDestroy(handle);
	}
};

}

#endif /* MATHEMATICS_OPT_CU_ */
