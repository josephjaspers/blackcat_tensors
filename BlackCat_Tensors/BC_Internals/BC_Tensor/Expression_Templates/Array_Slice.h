/*
 * Tensor_Slice.cu
 *
 *  Created on: Feb 25, 2018
 *      Author: joseph
 */

#ifndef TENSOR_SLICE_H_
#define TENSOR_SLICE_H_

#include "Array_Base.h"

namespace BC {
namespace internal {

//Floored decrement just returns the max(param - 1, 0)

template<class PARENT>
struct Array_Slice
		: Array_Base<Array_Slice<PARENT>, PARENT::DIMS() - 1>, Shape<PARENT::DIMS() - 1> {

	using scalar_t = typename PARENT::scalar_t;
	using mathlib_t = typename PARENT::mathlib_t;

	__BCinline__ static constexpr int ITERATOR() { return MTF::max(PARENT::ITERATOR() - 1, 0); }
	__BCinline__ static constexpr int DIMS() { return PARENT::DIMS() - 1; }

	scalar_t* array_slice;

	__BCinline__ Array_Slice(PARENT parent_, int index)
	: Shape<PARENT::DIMS() - 1> (parent_.as_shape()),
	  array_slice(const_cast<scalar_t*>(parent_.slice_ptr(index))) {}

	__BCinline__ const scalar_t* memptr() const { return array_slice; }
	__BCinline__	   scalar_t* memptr()   	{ return array_slice; }

	};

	template<class internal_t>
	auto make_slice(internal_t internal, int index) {
		return Array_Slice<internal_t>(internal, index);
	}
}
}
#endif /* TENSOR_SLICE_CU_ */
