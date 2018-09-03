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
		: Tensor_Array_Base<Array_Slice<PARENT>, MTF::max(PARENT::DIMS() - 1, 0)>,
		  Shape_Base<Array_Slice<PARENT>>{

	using scalar_t = typename PARENT::scalar_t;
	using mathlib_t = typename PARENT::mathlib_t;

	__BCinline__ static constexpr int ITERATOR() { return MTF::max(PARENT::ITERATOR() - 1, 0); }
	__BCinline__ static constexpr int DIMS() { return MTF::max(PARENT::DIMS() - 1, 0); }

	__BCinline__ operator const PARENT() const	{ return parent; }

	const PARENT parent;
	scalar_t* array_slice;

	__BCinline__ Array_Slice(const scalar_t* array, PARENT parent_) : array_slice(const_cast<scalar_t*>(array)), parent(parent_) {}

	__BCinline__ const auto inner_shape() const 			{ return parent.inner_shape(); }
	__BCinline__ const auto outer_shape() const 			{ return parent.outer_shape(); }
	__BCinline__ const auto block_shape() const 			{ return parent.block_shape(); }

	__BCinline__ int size() const { return parent.outer_shape()[DIMS() - 1]; }
	__BCinline__ int rows() const { return parent.inner_shape()[0]; }
	__BCinline__ int cols() const { return  parent.inner_shape()[1]; }
	__BCinline__ int dimension(int i) const { return parent.dimension(i); }
	__BCinline__ int outer_dimension() const { return parent.inner_shape()[DIMS() - 2]; }
	__BCinline__ int leading_dimension(int i) const { return DIMS() == 1 ? 1 : parent.leading_dimension(i); }

	__BCinline__ const scalar_t* memptr() const { return array_slice; }
	__BCinline__	   scalar_t* memptr()   	   { return array_slice; }

	};
}
}
#endif /* TENSOR_SLICE_CU_ */
