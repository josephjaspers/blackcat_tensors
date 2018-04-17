/*
 * Tensor_Slice.cu
 *
 *  Created on: Feb 25, 2018
 *      Author: joseph
 */

#ifndef TENSOR_Scalar_CU_
#define TENSOR_Scalar_CU_

#include "BlackCat_Tensor_Core_Base.h"

namespace BC {

template<class PARENT>
struct Tensor_Scalar : Tensor_Core_Base<Tensor_Scalar<PARENT>, 0> {

	using scalar = _scalar<PARENT>;

	const PARENT parent;
	scalar* array_slice;
	__BCinline__ static constexpr int CONTINUOUS() { return 0; }
	__BCinline__ Tensor_Scalar(const scalar* array, const PARENT& parent_)
		: array_slice(const_cast<scalar*>(array)), parent(parent_) {}

	__BCinline__ const auto innerShape() const 	{ return parent.innerShape(); }
	__BCinline__ const auto outerShape() const 	{ return parent.outerShape(); }

	__BCinline__ const scalar* getIterator() const { return array_slice; }
	__BCinline__	   scalar* getIterator()  	  { return array_slice; }

	};
}



#endif /* TENSOR_SLICE_CU_ */
