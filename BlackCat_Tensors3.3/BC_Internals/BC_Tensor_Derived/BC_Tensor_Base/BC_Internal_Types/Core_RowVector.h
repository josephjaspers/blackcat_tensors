/*
 * Core_RowVector.h
 *
 *  Created on: Mar 14, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CORE_ROWVECTOR_H_
#define TENSOR_CORE_ROWVECTOR_H_

#include "Core_Base.h"

namespace BC {
namespace internal {
/*
 * Accepts some core_type of Dimension 1 or 2 and returns a row_vector internal type
 */

template<class PARENT>
struct Tensor_Row : Tensor_Core_Base<Tensor_Row<PARENT>, 1>  {

	static_assert(PARENT::DIMS() == 2 || PARENT::DIMS() == 1, "TENSOR_ROW CAN ONLY BE GENERATED FROM ANOTHER VECTOR, ROW_VECTOR, OR MATRIX");

	using array = _iterator<PARENT>;
	using self = Tensor_Row<PARENT>;
	using slice_type = Tensor_Scalar<self>;
	using Mathlib = typename  PARENT::Mathlib;

	__BCinline__ static constexpr int DIMS() { return 1; }
	__BCinline__ static constexpr int ITERATOR() { return 0; }

	operator const PARENT() const	{ return parent; }

	PARENT parent;
	array array_slice;

	__BCinline__ Tensor_Row(array array, PARENT parent_) : array_slice(array), parent(parent_) {}
	__BCinline__ int increment() const { return parent.ld1(); }
	__BCinline__ int dims() const { return 1; }
	__BCinline__ int size() const { return parent.cols(); }
	__BCinline__ int rows() const { return parent.cols(); }
	__BCinline__ int cols() const { return 1; }
	__BCinline__ int dimension(int i) const { return i == 0 ? rows() : 1; }
	__BCinline__ int ld1() const { return parent.ld1(); }
	__BCinline__ int ld2() const { return 0; }
	__BCinline__ int LDdimension(int i) const { return i == 0 ? ld1() : 0; }
	__BCinline__ const auto inner_shape() const 			{ return parent.inner_shape(); }
	__BCinline__ const auto outer_shape() const 			{ return parent.outer_shape(); }

	__BCinline__ const auto& operator [] (int i) const { return array_slice[i * increment()]; }
	__BCinline__ 	   auto& operator [] (int i)  	   { return array_slice[i * increment()]; }

	void print_dimensions() 		const { parent.print_dimensions(); }
	void print_leading_dimensions()	const { parent.print_dimensions(); }

	__BCinline__ const auto slice(int i) const { return Tensor_Scalar<self>(&array_slice[i * increment()], *this); }
	__BCinline__	   auto slice(int i) 	   { return Tensor_Scalar<self>(&array_slice[i * increment()], *this); }


	__BCinline__ const auto& memptr() const { return *this; }
	__BCinline__	   auto& memptr()  	 { return *this; }

};
}
}


#endif /* TENSOR_CORE_ROWVECTOR_H_ */
