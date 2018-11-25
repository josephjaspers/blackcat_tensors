/*
 * Mathematics_CPU_Misc.h
 *
 *  Created on: May 6, 2018
 *      Author: scalar_toseph
 */

#ifndef MATHEMATICS_CPU_MISC_H_
#define MATHEMATICS_CPU_MISC_H_

#include <random>

namespace BC {
namespace evaluator {
namespace host_impl {
template<class core_lib>
struct Misc {

    template<typename T, typename scalar_t>
    static void randomize(T& tensor, scalar_t lower_bound, scalar_t upper_bound) {
 __BC_omp_for__
        for (int i = 0; i < tensor.size(); ++i) {
            tensor[i] = ((scalar_t) (std::rand() / ((scalar_t) RAND_MAX + 1)) * (upper_bound - lower_bound)) + lower_bound;
        }
 __BC_omp_bar__
    }

	template<class scalar_t>
    struct rand_handle {

    	scalar_t lower_bound;
    	scalar_t upper_bound;

    	rand_handle(scalar_t lb, scalar_t ub)
    		: lower_bound(lb), upper_bound(ub) {}

        scalar_t operator () () const {
           return (scalar_t(std::rand() / ((scalar_t)RAND_MAX)) * (upper_bound - lower_bound)) + lower_bound;
        }

        scalar_t operator [](unsigned i) {
            return (scalar_t(std::rand() / ((scalar_t)RAND_MAX)) * (upper_bound - lower_bound)) + lower_bound;
        }
    };

	template<class scalar_t>
	static auto make_rand_gen(scalar_t lower, scalar_t upper) {
		return rand_handle<scalar_t>(lower, upper);
	}
};
}
}
}

#endif /* MATHEMATICS_CPU_MISC_H_ */
