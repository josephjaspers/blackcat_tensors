/*
 * Array_Packed_Triangle.h
 *
 *  Created on: Sep 27, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_PACKED_TRIANGLE_H_
#define BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_PACKED_TRIANGLE_H_

#include "Array_Base.h"

namespace BC {
namespace internal {

enum uplo {
	up,
	lo
};

template<class scalar_t_, class mathlib_t_, uplo uplo_t = lo>
struct Packed_Triangle : Shape<2> {

	using scalar_t = scalar_t_;
	using mathlib_t = mathlib_t_;



	int data_length;
	int length;
	scalar_t* array;

	Packed_Triangle(int length) :
		Shape<2>(length, length),
		data_length(factorial_sum(length)),
		length(length) {

		mathlib_t::initialize(array, data_length);
	}



	int factorial_sum(int x) {
		return x == 1 ? 1 : x + factorial_sum(x-1);
	}
};
}
}



#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_PACKED_TRIANGLE_H_ */
