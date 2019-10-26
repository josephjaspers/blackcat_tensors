/*
 * Benchmark_MatMul_Reordering.h
 *
 *  Created on: Oct 23, 2018
 *      Author: joseph
 */

#ifndef BENCHMARK_MATMUL_REORDERING_H_
#define BENCHMARK_MATMUL_REORDERING_H_

#include "../include/BlackCat_Tensors.h"
#include <chrono>

float BC_matmul_assign(int size, BC::size_t  iters) {

    using mat = BC::Matrix<double>;

    mat a(size, size);
    mat b(size, size);
    mat c(size, size);

    b.randomize(-1000, 1000);
    c.randomize(-1000, 1000);

    auto wcts = std::chrono::system_clock::now();

    for (int i = 0; i < iters; ++i) {
        a = b * c;
    }

    std::chrono::duration<double> wctduration =
            (std::chrono::system_clock::now() - wcts);
    return wctduration.count();

}

float HC_matmul_assign(int size, BC::size_t  iters) {

    using mat = BC::Matrix<double>;

    mat a(size, size);
    mat b(size, size);
    mat c(size, size);
    mat d(size, size);

    b.randomize(-1000, 1000);
    c.randomize(-1000, 1000);
    d.randomize(-1000, 1000);

    double* a_ = a.data();
    double* b_ = b.data();
    double* c_ = c.data();

    auto wcts = std::chrono::system_clock::now();

    for (int i = 0; i < iters; ++i) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, size, size, size,
                1.0, b_, size, c_, size, 0, a_, size);
    }

    std::chrono::duration<double> wctduration =
            (std::chrono::system_clock::now() - wcts);
    return wctduration.count();


}
float BC_logistic_matmul_assign(int size, BC::size_t  iters) {

    using mat = BC::Matrix<double>;

    mat a(size, size);
    mat b(size, size);
    mat c(size, size);
    mat d(size, size);

    b.randomize(-1000, 1000);
    c.randomize(-1000, 1000);
    d.randomize(-1000, 1000);

    auto wcts = std::chrono::system_clock::now();

    for (int i = 0; i < iters; ++i) {
        a = BC::tanh(b * c + d);
    }

    std::chrono::duration<double> wctduration =
            (std::chrono::system_clock::now() - wcts);
    return wctduration.count();
}

float HC_logistic_matmul_assign(int size, BC::size_t  iters) {

    using mat = BC::Matrix<double>;

    mat a(size, size);
    mat b(size, size);
    mat c(size, size);
    mat d(size, size);

    b.randomize(-1000, 1000);
    c.randomize(-1000, 1000);
    d.randomize(-1000, 1000);

    double* a_ = a.data();
    double* b_ = b.data();
    double* c_ = c.data();
    double* d_ = d.data();

    auto wcts = std::chrono::system_clock::now();

    for (int i = 0; i < iters; ++i) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, size, size, size,
                1.0, b_, size, c_, size, 0, a_, size);

        for (int j = 0; j < size; ++j) {
            a_[j] = BC::tanh(a_[j] + d_[j]);
        }
    }
    std::chrono::duration<double> wctduration =
            (std::chrono::system_clock::now() - wcts);
    return wctduration.count();
}




void benchmark_matmul_suite(int count=10) {
	int size = 128;
	int reps = 100;
	float multiplier = 1.2;

	std::string markdown_header = {
		"|Size | BC time | Baseline | Performance difference |\n" \
		 "| --- | --- | --- | --- |"
	};



	std::cout << markdown_header << std::endl;
	for (int i = 0; i < count; ++i) {
		auto bc = BC_logistic_matmul_assign(size, reps);
		auto baseline = BC_logistic_matmul_assign(size, reps);

			std::cout << "|" << size \
						<< "|" << bc \
					       << "|" << baseline \
					          <<"|"<< baseline/bc << "|" << std::endl;

		size *= multiplier;
	}
}
#endif /* BENCHMARK_MATMUL_REORDERING_H_ */
