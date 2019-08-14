/*
 * c_ewise.h
 *
 *  Created on: Jan 21, 2019
 *      Author: joseph
 */

#ifndef BC_BENCHMARK_BC_EWISE_H_
#define BC_BENCHMARK_BC_EWISE_H_

#include "../include/BlackCat_Tensors.h"
#include "cu_cwise.cu"

namespace BC {
namespace benchmarks {

template<class scalar_t, class allocator=BC::Basic_Allocator<scalar_t>>
auto bc_cwise(int size, int reps) {

    using vec   = BC::Vector<scalar_t, allocator>;

    vec a(size);
    vec b(size);
    vec c(size);
    vec d(size);
    vec e(size);

    a.randomize(-1000, 1000);
    b.randomize(-1000, 1000);
    c.randomize(-1000, 1000);
    d.randomize(-1000, 1000);
    e.randomize(-1000, 1000);

	auto f = [&]() {
			a = b + c - d / e;
	};
	return timeit(f, reps);
}


}
}




#endif /* C_EWISE_H_ */
