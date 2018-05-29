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
#include "CPU_Implementation/CPU_BLAS.h"
#include "CPU_Implementation/CPU_Signal_Processing.h"

namespace BC {

/*
 * The core CPU library,
 *
 * defines:
 * 	copy(T, U, int)
 *
 * 	dimensional evaluations
 *
 */

class CPU:
		public CPU_Utility<CPU>,
		public CPU_Misc<CPU>,
		public CPU_BLAS<CPU>,
		public CPU_Signal_Processing<CPU> {


public:

	static constexpr int SINGLE_THREAD_THRESHOLD = 8192;

//-------------------------------Generic Copy ---------------------------------//
	template<typename T, typename J>
	static void copy(T& t, const J& j, int sz) {
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
	//-------------------------------1d eval/copy ---------------------------------//

	struct n0 {
		template<class T>
		static void eval(T to) {
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < to.size(); ++i) {
				to[i];
			}

#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
		}
	};

	//-------------------------------2d eval/copy ---------------------------------//
	struct n2 {
		template<class T>
		static void eval(T to) {
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
			for (int n = 0; n < to.dimension(1); ++n)
				for (int m = 0; m < to.dimension(0); ++m)
					to(m, n);

#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
		}
	};
	//-------------------------------3d eval/copy ---------------------------------//

	struct n3 {
		template<class T>
		static void eval(T to) {
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
			for (int k = 0; k < to.dimension(2); ++k)
				for (int n = 0; n < to.dimension(1); ++n)
					for (int m = 0; m < to.dimension(0); ++m)
						to(m, n, k);

#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
		}
	};
	//-------------------------------4d eval/copy ---------------------------------//4
	struct n4 {
		template<class T>
		static void eval(T to) {
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
			for (int p = 0; p < to.dimension(3); ++p)
				for (int k = 0; k < to.dimension(2); ++k)
					for (int n = 0; n < to.dimension(1); ++n)
						for (int m = 0; m < to.dimension(0); ++m)
							to(m, n, k, p);
#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
		}
	};
	//-------------------------------5d eval/copy ---------------------------------//

	struct n5 {
		template<class T>
		static void eval(T to) {
#ifndef BC_NO_OPENMP
#pragma omp parallel for
#endif
			for (int j = 0; j < to.dimension(4); ++j)
				for (int p = 0; p < to.dimension(3); ++p)
					for (int k = 0; k < to.dimension(2); ++k)
						for (int n = 0; n < to.dimension(1); ++n)
							for (int m = 0; m < to.dimension(0); ++m)
								to(m, n, k, p, j);
#ifndef BC_NO_OPENMP
#pragma omp barrier
#endif
		}
	};
	//-------------------------------implementation ---------------------------------//

	template<int d>
	struct dimension {
		using run = std::conditional_t<(d <= 1), n0,
		std::conditional_t< d ==2, n2,
		std::conditional_t< d == 3, n3,
		std::conditional_t< d == 4, n4, n5>>>>;

		template<class T>
		static void eval(T to) {
			run::eval(to);
		}
	};
};
}
#endif /* MATHEMATICS_CPU_H_ */
