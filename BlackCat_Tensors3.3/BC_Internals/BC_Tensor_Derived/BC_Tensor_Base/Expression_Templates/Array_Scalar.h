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

	using scalar = _scalar<PARENT>;

	__BCinline__ static constexpr int ITERATOR() { return 0; }
	__BCinline__ static constexpr int DIMS() { return 0; }

	__BCinline__ const auto& operator [] (int index) const { return array_slice[0]; }
	__BCinline__ 	   auto& operator [] (int index) 	   { return array_slice[0]; }

	template<class ... integers> __BCinline__ auto& operator ()(integers ... ints) {
		return array_slice[0];
	}
	template<class ... integers> __BCinline__ const auto& operator ()(integers ... ints) const {
		return array_slice[0];
	}

	__BCinline__ operator const PARENT() const	{ return parent; }

	const PARENT parent;
	scalar* array_slice;

	__BCinline__ Array_Scalar(const scalar* array, const PARENT& parent_)
		: array_slice(const_cast<scalar*>(array)), parent(parent_) {}

	__BCinline__ const scalar* memptr() const { return array_slice; }
	__BCinline__	   scalar* memptr()  	  { return array_slice; }
};
}
}



#endif /* TENSOR_SLICE_CU_ */
