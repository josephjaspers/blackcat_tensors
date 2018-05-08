/*
 * BC_Mathematics_CPU.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef MATHEMATICS_CPU_H_
#define MATHEMATICS_CPU_H_

#include "Print.h"
#include "CPU_Implementation/CPU_Utility.h"
#include "CPU_Implementation/CPU_Misc.h"
#include "CPU_Implementation/CPU_BLAS.h"

namespace BC {

/*
 * The core CPU library,
 * defines the iterators for Tensors
 */

class CPU
		: public CPU_Utility<CPU>,
		  public CPU_Misc<CPU>,
		  public CPU_BLAS<CPU>	{

public:

	static constexpr int SINGLE_THREAD_THRESHOLD = 8192;



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
	struct n0 {
		template<class T, class U>
		static void copy(T to, U from) {
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < to.size(); ++i) {
				to[i] = from[i];
			}
		}
#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
	};
	struct n2 {
		template<class T, class U>
		static void copy(T to, U from) {
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
			for (int n = 0; n < to.dimension(1); ++n)
				for (int m = 0; m < to.dimension(0); ++m)
					to(m,n) = from(m,n);
		}
#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
	};
	struct n3 {

		template<class T, class U>
		static void copy(T to, U from) {
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
			for (int k = 0; k < to.dimension(2); ++k)
				for (int n = 0; n < to.dimension(1); ++n)
					for (int m = 0; m < to.dimension(0); ++m)
						to(m,n,k) = from(m,n,k);
		}
#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
	};

	struct n4 {
		template<class T, class U>
		static void copy(T to, U from) {
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
			for (int p = 0; p < to.dimension(3); ++p)
				for (int k = 0; k < to.dimension(2); ++k)
					for (int n = 0; n < to.dimension(1); ++n)
						for (int m = 0; m < to.dimension(0); ++m)
							to(m, n, k, p) = from(m, n, k, p);
		}
#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
	};
	struct n5 {
		template<class T, class U>
		static void copy(T to, U from) {
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
			for (int j = 0; j < to.dimension(4); ++j)
				for (int p = 0; p < to.dimension(3); ++p)
					for (int k = 0; k < to.dimension(2); ++k)
						for (int n = 0; n < to.dimension(1); ++n)
							for (int m = 0; m < to.dimension(0); ++m)
								to(m, n, k, p, j) = from(m, n, k, p, j);
		}
#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
	};

	template<int d>
	struct dimension {
		using run = std::conditional_t<(d <= 1), n0,
						std::conditional_t< d ==2, n2,
							std::conditional_t< d == 3, n3,
								std::conditional_t< d == 4, n4, n5>>>>;
		template<class T, class U>
		static void copy(T to, U from) {
			run::copy(to, from);
		}
	};











};
}
#endif /* MATHEMATICS_CPU_H_ */

//static constexpr int max(int a, int b) {
//	return a > b ? a : b;
//}
//
//struct v1 {
//	template<class T, class U, class ... integers>
//	static void run(T to, U from, integers ... ints) {
//		for (int i = 0; i < from.dimension(d - 1); ++i)
//			dimension<max(d - 1, 0)>::copy(to, from, i, ints...);
//	}
//};
//struct v2 {
//	template<class T, class U, class ... integers>
//	static void run(T to, U from, integers ... ints) {
//		for (int i = 0; i < from.dimension(0); ++i)
//			to(i, ints...) = from(i, ints...);
//	}
//};
