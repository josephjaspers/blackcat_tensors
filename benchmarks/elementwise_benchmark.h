/*
 * elementwise_benchmark.h
 *
 *  Created on: Jan 21, 2019
 *      Author: joseph
 */

#ifndef BC_ELEMENTWISE_BENCHMARK_H_
#define BC_ELEMENTWISE_BENCHMARK_H_

#include <iostream>

#include "common.h"
#include "coefficientwise/BC_cwise.h"
#include "coefficientwise/c_cwise.h"

#ifdef __CUDACC__
#include "coefficientwise/cu_cwise.cu"
#endif

namespace BC {
namespace benchmarks {


auto cwise_benchmark(int sz, int reps=10, bool stdout=false) {

	auto bc_time = bc_cwise<float, BC::Basic_Allocator<float>>(sz, reps);
	auto c_time  =  c_cwise<float, std::allocator<float>>(sz, reps);

#ifdef __CUDACC__
	auto bc_cu_time = bc_cwise<float, BC::Cuda<float>>(sz, reps);
	cudaDeviceSynchronize();
	auto cu_time = cu_cwise<float, BC::Cuda<float>>(sz, reps);
	cudaDeviceSynchronize();

	if (stdout) {
		std::cout << "bc_time: " << bc_time \
				<< "\nc_time: " << c_time \
				<< "\nbc_cu_time: " << bc_cu_time \
				<< "\ncu_time " << cu_time << std::endl;
	}

	return std::make_tuple(bc_time, c_time, cu_time);

#else
	if (stdout) {
		std::cout << "bc_time: " << bc_time \
				<< "\nc_time: " << c_time << std::endl;
	}


	return std::make_tuple(bc_time, c_time);
#endif

}




}
}




#endif /* ELEMENTWISE_BENCHMARK_H_ */
