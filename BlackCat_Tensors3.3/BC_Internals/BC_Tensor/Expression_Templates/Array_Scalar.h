/*
 * Tensor_Slice.cu
 *
 *  Created on: Feb 25, 2018
 *      Author: joseph
 */

#ifndef TENSOR_SCALAR_H_
#define TENSOR_SCALAR_H_

#include "Array_Base.h"

namespace BC {
namespace internal {
/*
 * Represents a single_scalar value from a tensor
 */

template<class PARENT>
struct Array_Scalar : Tensor_Array_Base<Array_Scalar<PARENT>, 0>, Shape<0> {

	using scalar_t = typename PARENT::scalar_t;
	using mathlib_t = typename PARENT::mathlib_t;

	__BCinline__ static constexpr int ITERATOR() { return 0; }
	__BCinline__ static constexpr int DIMS() { return 0; }

	scalar_t& array_slice;

	__BCinline__ Array_Scalar(const scalar_t& array) : array_slice(const_cast<scalar_t&>(array)) {}

	__BCinline__ const auto& operator [] (int index) const { return array_slice; }
	__BCinline__ 	   auto& operator [] (int index) 	   { return array_slice; }

	template<class... integers> __BCinline__
	auto& operator ()(integers ... ints) {
		return array_slice[0];
	}
	template<class... integers> __BCinline__
	const auto& operator ()(integers ... ints) const {
		return array_slice[0];
	}

	__BCinline__ const scalar_t* memptr() const { return &array_slice; }
	__BCinline__	   scalar_t* memptr()  	  { return &array_slice; }
};
}
}



#endif /* TENSOR_SLICE_CU_ */
