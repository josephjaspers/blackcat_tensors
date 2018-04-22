/*
 * BC_Mathematics_CPU.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef MATHEMATICS_CPU_H_
#define MATHEMATICS_CPU_H_

#include <cmath>
#include <iostream>
#include <string>

#include "BC_PrintFunctions.h"
#include <cblas.h>

namespace BC {

class CPU {

	static constexpr int SINGLE_THREAD_THRESHOLD = 8192;

public:
	template<typename T>
	static T*& initialize(T*& t, int sz) {
		t = new T[sz];
		return t;
	}

	template<typename T>
	static T*& unified_initialize(T*& t, int sz) {
		t = new T[sz];
		return t;
	}

	//The tensor classes mandate these methods differentiate method calls between gpu and cpu libraries, ergo you must override certain methods.
	template<class T, class U>
	static void HostToDevice(T* t, U* u, int size) {
		copy(t, u, size);
	}
	template<class T, class U>
	static void DeviceToHost(T* t, U* u, int size) {
		copy(t, u, size);
	}
	template<typename T>
	static void destroy(T* t) {
		delete[] t;
	}
	template<typename T, typename J>
	static void fill(T& t, const J j, int sz) {
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sz; ++i) {
			t[i] = j;
		}
#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
	}
	template<typename T>
	static void zero(T& t, int sz) {
		fill(t, 0, sz);
	}
	template<typename T, typename J> __attribute__((always_inline)) inline
	static void copy(T& t, const J& j, int sz) {
		if (sz < SINGLE_THREAD_THRESHOLD) {
			for (int i = 0; i < sz; ++i) {
				t[i] = j[i];
			}
			return;
		}
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sz; ++i) {
			t[i] = j[i];
		}
#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
	}


	template<int d>
	struct dimension {
		static constexpr int max(int a, int b) {
			return a > b ? a : b;
		}

		struct v1 {
			template<class T, class U, class ... integers>
			static void run(T to, U from, integers ... ints) {
				for (int i = 0; i < from.dimension(d - 1); ++i)
					dimension<max(d - 1, 0)>::copy(to, from, i, ints...);
			}
		};
		struct v2 {
			template<class T, class U, class ... integers>
			static void run(T to, U from, integers ... ints) {
				for (int i = 0; i < from.dimension(0); ++i)
					to(i, ints...) = from(i, ints...);
			}
		};

		template<class T, class U, class ... integers>
		static void copy(T to, U from, integers ... ints) {

			using runner = std::conditional_t<(d > 1), v1, v2>;
			runner::run(to, from, ints...);
		}
	};


//	template<typename T, typename J> __attribute__((always_inline)) inline static void copy1d(T& t, const J& j) { return copy(t, j, t.size()); }
//	template<typename T, typename J> __attribute__((always_inline)) inline static void copy2d(T& t, const J& j) {
//			for (int n = 0; n < j.cols(); ++n)
//				for (int m = 0; m < j.rows(); ++m)
//					t(m,n) = j(m,n);
//	}
//	template<typename T, typename J> __attribute__((always_inline)) inline static void copy3d(T& t, const J& j) {
//		for (int k = 0; k < j.dimension(2); ++k)
//			for (int n = 0; n < j.cols(); ++n)
//				for (int m = 0; m < j.rows(); ++m)
//					t(m,n,k) = j(m,n,k);
//	}
//	template<typename T, typename J> __attribute__((always_inline)) inline static void copy4d(T& t, const J& j) {
//		for (int l = 0; l < j.dimension(3); ++l)
//			for (int k = 0; k < j.dimension(2); ++k)
//				for (int n = 0; n < j.cols(); ++n)
//					for (int m = 0; m < j.rows(); ++m)
//						t(m,n,k,l) = j(m,n,k,l);
//	}

	template<typename T, typename J>
	static void randomize(T& t, J lower_bound, J upper_bound, int sz) {
		if (sz < SINGLE_THREAD_THRESHOLD) {
			for (int i = 0; i < sz; ++i) {
				t[i] = ((double) (rand() / ((double) RAND_MAX + 1)) * (upper_bound - lower_bound)) + lower_bound;
			}
			return;
		}
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sz; ++i) {
			t[i] = ((double) (rand() / ((double) RAND_MAX + 1)) * (upper_bound - lower_bound)) + lower_bound;
		}
#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
	}


	template<class T, class RANKS>
	static void print(const T ary, const RANKS ranks, int order, int print_length) {
		BC::print(ary, ranks, order, print_length);
	}

	template<class T, class RANKS>
	static void printSparse(const T ary, const RANKS ranks, int order, int print_length) {
		BC::printSparse(ary, ranks, order, print_length);
	}



	/*
	 * a = M x K
	 * b = K x N
	 * c = M x N
	 */

	//
public:
	template<class U, class T, class V>
	static void scalarMul(U eval, T a, V b) {
		*eval = a[0] * b[0];
	}

	static void MatrixMul(bool transA, bool transB, const float* A, const float* B, float* C, int m, int n, int k,
			const float* scalarA = nullptr, const float* scalarB = nullptr,  int lda = 0, int ldb =0, int ldc =0) {

		 auto TRANS_A =  transA ? CblasTrans : CblasNoTrans;
		 auto TRANS_B =  transB ? CblasTrans : CblasNoTrans;

		  if (lda == 0 ) lda = m;
		  if (ldb == 0 ) ldb = n;
		  if (ldc == 0 ) ldc = m;

	      const float beta   =  scalarB ? *scalarB : 0;
		  const float alpha  =  scalarA ? *scalarA : 1;

	{
		cblas_sgemm(CblasColMajor, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	}
	static void MatrixMul(bool transA, bool transB, const double* A, const double* B, double* C, int m, int n, int k,
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
#endif /* MATHEMATICS_CPU_H_ */
