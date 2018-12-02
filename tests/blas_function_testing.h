/*
 * blas_function_testing.h
 *
 *  Created on: Dec 1, 2018
 *      Author: joseph
 */

#ifndef BLAS_FUNCTION_TESTING_H_
#define BLAS_FUNCTION_TESTING_H_

#include "../include/BlackCat_Tensors.h"

template<class scalar_t=float, class alloc_t=BC::Basic_Allocator>
int blas_function_testing(int size=128) {

	using mat = BC::Matrix<scalar_t, alloc_t>;

	mat a(size, size);
	mat b(size, size);
	mat y(size, size);

	y = a * b;



}




#endif /* BLAS_FUNCTION_TESTING_H_ */
