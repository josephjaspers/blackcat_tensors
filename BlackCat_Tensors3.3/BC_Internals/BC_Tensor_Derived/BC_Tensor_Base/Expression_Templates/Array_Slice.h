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
	struct Array_Slice : Tensor_Array_Base<Array_Slice<PARENT>, MTF::max(PARENT::DIMS() - 1, 0)> {

	using scalar_type = _scalar<PARENT>;

	__BCinline__ static constexpr int ITERATOR() { return MTF::max(PARENT::ITERATOR() - 1, 0); }
	__BCinline__ static constexpr int DIMS() { return MTF::max(PARENT::DIMS() - 1, 0); }

	__BCinline__ operator const PARENT() const	{ return parent; }

	const PARENT parent;
	scalar_type* array_slice;

	__BCinline__ Array_Slice(const scalar_type* array, PARENT parent_) : array_slice(const_cast<scalar_type*>(array)), parent(parent_) {}

	__BCinline__ const auto inner_shape() const 			{ return parent.inner_shape(); }
	__BCinline__ const auto outer_shape() const 			{ return parent.outer_shape(); }
	__BCinline__ int size() const { return parent.outer_shape()[DIMS()]; }
	__BCinline__ int rows() const { return parent.inner_shape()[0]; }
	__BCinline__ int cols() const { return  parent.inner_shape()[1]; }
	__BCinline__ int dimension(int i) const { return parent.dimension(i); }
	__BCinline__ int outer_dimension() const { return parent.inner_shape()[DIMS() - 2]; }
	__BCinline__ int leading_dimension(int i) const { return parent.leading_dimension(i); }

	__BCinline__ const scalar_type* memptr() const { return array_slice; }
	__BCinline__	   scalar_type* memptr()   	   { return array_slice; }

	};
}
}
#endif /* TENSOR_SLICE_CU_ */
