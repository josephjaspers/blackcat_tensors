/*
 * Tensor_Core_RowVector.h
 *
 *  Created on: Mar 14, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CORE_ROWVECTOR_H_
#define TENSOR_CORE_ROWVECTOR_H_

#include "BC_Expressions/Expression_Base.h"
#include "Determiners.h"
#include "Tensor_Core_Scalar.h"

namespace BC {

template<class PARENT>
struct Tensor_Row : expression<_scalar<PARENT>, Tensor_Row<PARENT>>  {

	using scalar = _scalar<PARENT>;
	using self = Tensor_Row<PARENT>;
	using slice_type = Tensor_Scalar<self>;

	static constexpr int DIMS() { return 1; }
	static_assert(PARENT::DIMS() == 2 || PARENT::DIMS() == 1, "TENSOR_ROW CAN ONLY BE GENERATED FROM ANOTHER VECTOR, ROW_VECTOR, OR MATRIX");

	const PARENT parent;
	scalar* array_slice;

	operator 	   scalar*()       { return array_slice; }
	operator const scalar*() const { return array_slice; }

	Tensor_Row(scalar* array, const PARENT& parent_) : array_slice(array), parent(parent_) {}
	__BCinline__ int increment() const { return parent.LD_rows(); }
	__BCinline__ int dims() const { return 1; }
	__BCinline__ int size() const { return parent.cols(); }
	__BCinline__ int rows() const { return parent.cols(); }
	__BCinline__ int cols() const { return 1; }
	__BCinline__ int dimension(int i) const { return i == 0 ? rows() : 1; }
	__BCinline__ int LD_rows() const { return parent.LD_rows(); }
	__BCinline__ int LD_cols() const { return 0; }
	__BCinline__ int LDdimension(int i) const { return i == 0 ? LD_rows() : 0; }
	__BCinline__ const auto innerShape() const 			{ return parent.innerShape(); }
	__BCinline__ const auto outerShape() const 			{ return parent.outerShape(); }

	__BCinline__ const auto& operator [] (int i) const { return array_slice[i * increment()]; }
	__BCinline__ 	   auto& operator [] (int i)  	   { return array_slice[i * increment()]; }

	void printDimensions() 		const { parent.printDimensions(); }
	void printLDDimensions()	const { parent.printDimensions(); }

	const auto slice(int i) const { return Tensor_Scalar<self>(&array_slice[i * increment()], *this); }
		  auto slice(int i) 	  { return Tensor_Scalar<self>(&array_slice[i * increment()], *this); }


	const scalar* core() const { return array_slice; }
		  scalar* core()  	   { return array_slice; }

};

}


#endif /* TENSOR_CORE_ROWVECTOR_H_ */
