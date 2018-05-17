/*
 * Mathematics_CPU_Misc.h
 *
 *  Created on: May 6, 2018
 *      Author: joseph
 */

#ifndef MATHEMATICS_CPU_MISC_H_
#define MATHEMATICS_CPU_MISC_H_

namespace BC {

/*
 * Defines:
 *
 * 	randomize
 * 	fill
 * 	zero
 *
 */

template<class core_lib>
struct CPU_Misc {

	template<typename T, typename J>
	static void randomize(T& t, J lower_bound, J upper_bound, int sz) {
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

};

}

#endif /* MATHEMATICS_CPU_MISC_H_ */
