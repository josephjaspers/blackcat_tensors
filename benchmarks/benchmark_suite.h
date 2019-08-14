/*
 * benchmark_suite.h
 *
 *  Created on: Jan 22, 2019
 *      Author: joseph
 */

#ifndef BENCHMARK_SUITE_H_
#define BENCHMARK_SUITE_H_

#include "coefficientwise_benchmark.h"
#include <vector>


namespace BC {
namespace benchmarks {

//Requires Cuda to run

void run(std::vector<int> sizes = {10, 50, 100, 500, 1000, 2500, 5000, 75000, 10000, 20000, 40000 }, int reps=100000) {
#ifndef __CUDACC__
	std::cout << "| Size | C-impl | BC_host | Host-performance |\n";
	std::cout << "| ---  | --- | --- | --- | \n";

	for (int sz : sizes) {
		auto times = cwise_benchmark(sz, reps, false);

		std::cout << "|" << sz <<
				"|" << std::get<0>(times) << "|" << std::get<1>(times) << "|" << std::get<0>(times)/std::get<1>(times) << std::endl;	}
#else
	std::cout << "| Size | C-impl | BC_host | Host-performance | Cuda-impl | BC_device | Device-Performance | \n";
	std::cout << "| ---  | --- | --- | --- | --- | --- | --- | \n";

	for (int sz : sizes) {
		auto times = cwise_benchmark(sz, reps, false);

		std::cout << "|" << sz <<
				"|" << std::get<0>(times) << "|" << std::get<1>(times) << "|" << std::get<0>(times)/std::get<1>(times) << \
				"|" << std::get<2>(times) << "|" << std::get<3>(times) << "|" << std::get<2>(times)/std::get<3>(times) <<  "|" <<std::endl;
	}
#endif
}


}
}


#endif /* BENCHMARK_SUITE_H_ */
