/*
 * Tensor_Slice.cu
 *
 *  Created on: Feb 25, 2018
 *      Author: joseph
 */

#ifndef TENSOR_Scalar_CU_
#define TENSOR_Scalar_CU_

#include "BC_Expressions/Expression_Base.h"
#include "Determiners.h"

namespace BC {

template<class PARENT>
struct Tensor_Scalar : expression<_scalar<PARENT>, Tensor_Scalar<PARENT>> {

	using scalar = _scalar<PARENT>;
	using self = Tensor_Scalar<PARENT>;


	static constexpr int DIMS() { return 0; }

	const PARENT parent;
	scalar* array_slice;

	operator 	   scalar*()       { return array_slice; }
	operator const scalar*() const { return array_slice; }

	Tensor_Scalar(scalar* array, const PARENT& parent_) : array_slice(array), parent(parent_) {}

	__BCinline__ int dims() const { return 0; }
	__BCinline__ int size() const { return 1; }
	__BCinline__ int rows() const { return 1; }
	__BCinline__ int cols() const { return 1; }
	__BCinline__ int dimension(int i) const { return 1; }
	__BCinline__ int LD_rows() const { return 0; }
	__BCinline__ int LD_cols() const { return 0; }
	__BCinline__ int LDdimension(int i) const { return 0; }
	__BCinline__ const auto innerShape() const 			{ return parent.innerShape(); }
	__BCinline__ const auto outerShape() const 			{ return parent.outerShape(); }

	__BCinline__ const auto& operator [] (int i) const { return array_slice[0]; }
	__BCinline__ 	   auto& operator [] (int i)  	   { return array_slice[0]; }

	void printDimensions() 		const { parent.printDimensions(); }
	void printLDDimensions()	const { parent.printDimensions(); }

	const scalar* core() const { return array_slice; }
		  scalar* core()  	   { return array_slice; }


	};
}



#endif /* TENSOR_SLICE_CU_ */
