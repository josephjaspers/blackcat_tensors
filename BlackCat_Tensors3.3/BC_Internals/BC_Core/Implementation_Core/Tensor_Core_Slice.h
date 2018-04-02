/*
 * Tensor_Slice.cu
 *
 *  Created on: Feb 25, 2018
 *      Author: joseph
 */

#ifndef TENSOR_SLICE_CU_
#define TENSOR_SLICE_CU_

#include "BC_Expressions/Expression_Base.h"
#include "Determiners.h"
#include <iostream>
#include "Tensor_Core_Scalar.h"
#include "Tensor_Core_RowVector.h"
namespace BC {

template<class PARENT>
	struct Tensor_Slice : expression<_scalar<PARENT>, Tensor_Slice<PARENT>> {

	using scalar_type = _scalar<PARENT>;
	using self = Tensor_Slice<PARENT>;
	using Mathlib = typename  PARENT::Mathlib;
	__BCinline__ static constexpr int DIMS() { return PARENT::DIMS() - 1 > 0 ? PARENT::DIMS() - 1 : 0; };
	__BCinline__ static constexpr int LAST()  { return  PARENT::LAST() - 1; }
	using slice_type = std::conditional_t<DIMS() == 0, self, Tensor_Slice<self>>;

	const PARENT parent;
	scalar_type* array_slice;

	operator const PARENT	   () const	{ return parent;	  }
	operator 	   scalar_type*()       { return array_slice; }
	operator const scalar_type*() const { return array_slice; }

	__BCinline__ Tensor_Slice(scalar_type* array, PARENT parent_) : array_slice(array), parent(parent_) {}

	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int size() const { return DIMS() > 0 ? parent.outerShape()[LAST()] : 1;    }
	__BCinline__ int rows() const { return DIMS() > 0 ? parent.innerShape()[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? parent.innerShape()[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? parent.innerShape()[i] : 1; }
	__BCinline__ int LD_rows() const { return DIMS() > 0 ? parent.outerShape()[0] : 1; }
	__BCinline__ int LD_cols() const { return DIMS() > 1 ? parent.outerShape()[1] : 1; }
	__BCinline__ int LDdimension(int i) const { return DIMS() > i + 1 ? parent.outerShape()[i] : 1; }
	__BCinline__ const auto& operator [] (int i) const { return DIMS() == 0 ? array_slice[0] : array_slice[i]; }
	__BCinline__ auto& operator [] (int i)  	       { return DIMS() == 0 ? array_slice[0] : array_slice[i]; }

	void printDimensions() 		const { parent.printDimensions(); }
	void printLDDimensions()	const { parent.printDimensions(); }

	__BCinline__ const auto innerShape() const 			{ return parent.innerShape(); }
	__BCinline__ const auto outerShape() const 			{ return parent.outerShape(); }

	__BCinline__
	int slice_index(int i) const {
		if (DIMS() == 0)
			return 0;
		else if (DIMS() == 1)
			return i;
		else
			return outerShape()[LAST() - 1] * i;
	}
	__BCinline__ const auto slice(int i) const { return slice_type(&array_slice[slice_index(i)],*this); }
	__BCinline__	   auto slice(int i) 	   { return slice_type(&array_slice[slice_index(i)],*this); }

	__BCinline__ const auto scalar(int i) const { return Tensor_Scalar<self>(&array_slice[i], *this); }
	__BCinline__ auto scalar(int i) { return Tensor_Scalar<self>(&array_slice[i], *this); }

	__BCinline__ const auto row(int i) const { return Tensor_Row<self>(&array_slice[i], *this); }
	__BCinline__ auto row(int i) { return Tensor_Row<self>(&array_slice[i], *this); }

	__BCinline__ const scalar_type* getIterator() const { return array_slice; }
	__BCinline__	   scalar_type* getIterator()   	{ return array_slice; }



	};


}



#endif /* TENSOR_SLICE_CU_ */
