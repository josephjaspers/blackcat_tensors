/*
 * common.h
 *
 *  Created on: Jan 21, 2019
 *      Author: joseph
 */

#ifndef BC_BENCHMARKS_COMMON_H_
#define BC_BENCHMARKS_COMMON_H_

namespace BC {
namespace benchmarks {

template<class function>
float timeit (function& func, int iters=10) {

    using clock = std::chrono::duration<double>;

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < iters; ++i) {
    	func();
    }

    auto end = std::chrono::system_clock::now();
    clock total = clock(end - start);
    return total.count();
}
}
}


#endif /* COMMON_H_ */
