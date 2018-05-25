/*
 * Tensor_Slice.cu
 *
 *  Created on: Feb 25, 2018
 *      Author: joseph
 */

#ifndef TENSOR_SCALAR_H_
#define TENSOR_SCALAR_H_

#include "Core_Base.h"

namespace BC {

/*
 * Represents a single_scalar value from a tensor
 */

template<class PARENT>
struct Tensor_Scalar : Tensor_Core_Base<Tensor_Scalar<PARENT>, 0> {

	using scalar = _scalar<PARENT>;

	__BCinline__ static constexpr int ITERATOR() { return 0; }
	__BCinline__ static constexpr int DIMS() { return 0; }

	operator const PARENT() const	{ return parent; }

	const PARENT parent;
	scalar* array_slice;

	__BCinline__ Tensor_Scalar(const scalar* array, const PARENT& parent_)
		: array_slice(const_cast<scalar*>(array)), parent(parent_) {}

	__BCinline__ const auto innerShape() const 	{ return parent.innerShape(); }
	__BCinline__ const auto outerShape() const 	{ return parent.outerShape(); }

	__BCinline__ const scalar* getIterator() const { return array_slice; }
	__BCinline__	   scalar* getIterator()  	  { return array_slice;  }
	};
}



#endif /* TENSOR_SLICE_CU_ */
