/*
 * Mathematics_CPU_Misc.h
 *
 *  Created on: May 6, 2018
 *      Author: scalar_toseph
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

	template<typename T, typename scalar_t>
	static void randomize(T& tensor, scalar_t lower_bound, scalar_t upper_bound) {
 __BC_omp_for__
		for (int i = 0; i < tensor.size(); ++i) {
			tensor[i] = ((scalar_t) (std::rand() / ((scalar_t) RAND_MAX + 1)) * (upper_bound - lower_bound)) + lower_bound;
		}
 __BC_omp_bar__
	}


	template<class scalar_t>
	struct rand_t {

		struct rand_handle {
			scalar_t operator () (scalar_t lower_bound, scalar_t upper_bound) const {
				return ((scalar_t) (std::rand() / ((scalar_t) RAND_MAX + 1)) * (upper_bound - lower_bound)) + lower_bound;
			}
		};

		scalar_t min;
		scalar_t max;

		rand_handle rand_handle_obj;
		rand_t(scalar_t min_, scalar_t max_) : min(min_), max(max_) {}
		__BCinline__ auto operator () (scalar_t v) const {

			return rand_handle_obj(min,max);
		}
	};

};

}

#endif /* MATHEMATICS_CPU_MISC_H_ */
