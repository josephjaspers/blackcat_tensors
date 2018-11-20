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
};

}

#endif /* MATHEMATICS_CPU_MISC_H_ */
