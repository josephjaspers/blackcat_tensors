/*
 * Elementwise.h
 *
 *  Created on: Oct 23, 2018
 *      Author: joseph
 */

#ifndef ELEMENTWISE_H_
#define ELEMENTWISE_H_
#include <vector>
#include <chrono>
#include "../../BlackCat_Tensors/include/BlackCat_Tensors.h"

template<class Allocator>
auto benchmark_forloop(int size, BC::size_t  iters, Allocator alloc, bool stdout=false) {

    using clock = std::chrono::duration<double>;
    using vec = BC::Vector<typename Allocator::value_type, Allocator>;

    clock bc_time;
    clock cptr_time;

    vec a(size);
    vec b(size);
    vec c(size);
    vec d(size);
    vec e(size);

    b.randomize(-1000, 1000);
    c.randomize(-1000, 1000);
    d.randomize(-1000, 1000);
    e.randomize(-1000, 1000);

    double* a_ = a.data();
    double* b_ = b.data();
    double* c_ = c.data();
    double* d_ = d.data();
    double* e_ = e.data();

    {
        auto wcts = std::chrono::system_clock::now();

        for (int i = 0; i < iters; ++i)
            a = b + c - d / e;

        bc_time = clock(std::chrono::system_clock::now() - wcts);
        
	    if (stdout)
                std::cout << "BlackCat_Expression a = b + c - d / e \n" << bc_time.count() << " seconds [Wall Clock]\n\n" << std::endl;
    }

    {
        auto wcts = std::chrono::system_clock::now();

        for (int i = 0; i < iters; ++i)
#pragma omp parallel for
            for (int j = 0; j < size; ++j) {
                a_[j] = b_[j] + c_[j] - d_[j] / e_[j];
            }
#pragma omp barrier
        cptr_time = clock(std::chrono::system_clock::now() - wcts);
        
	if (stdout)
            std::cout << "For Loop  a = b + c - d / e \n" << cptr_time.count() << " seconds [Wall Clock]\n\n" << std::endl;

    }

    return std::make_pair(bc_time, cptr_time);

}


template<class allocator=BC::Allocator<BC::host_tag, double>>
void benchmark_forloop_suite(bool stdout=false, allocator alloc=allocator()) {
	int size = 1000000;
	int reps = 100;
	float multiplier = 1.5;

	std::string markdown_header = {
		"|Size | BC time | Baseline | Performance difference |\n" \
		 "| --- | --- | --- | --- |"
	};



	std::cout << markdown_header << std::endl;
	for (int i = 0; i < 10; ++i) {
		auto times = benchmark_forloop(size, reps, alloc, stdout);
		auto bc = times.first;
		auto baseline = times.second;

			std::cout << "|" << size \
						<< "|" << bc.count() \
					       << "|" << baseline.count() \
					          <<"|"<< baseline.count()/bc.count() << "|" << std::endl;

		size *= multiplier;
	}
}


#endif /* ELEMENTWISE_H_ */
