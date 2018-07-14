/*
 * Array_RowVector.h
 *
 *  Created on: Mar 14, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CORE_ROWVECTOR_H_
#define TENSOR_CORE_ROWVECTOR_H_

#include "Array_Base.h"

namespace BC {
namespace internal {
/*
 * Accepts some core_type of Dimension 1 or 2 and returns a row_vector internal type
 */

template<class PARENT>
struct Array_Row : Tensor_Array_Base<Array_Row<PARENT>, 1>  {

	static_assert(PARENT::DIMS() == 2 || PARENT::DIMS() == 1, "TENSOR_ROW CAN ONLY BE GENERATED FROM ANOTHER VECTOR, ROW_VECTOR, OR MATRIX");

	using array = _iterator<PARENT>;
	using self = Array_Row<PARENT>;

	__BCinline__ static constexpr int DIMS() { return 1; }
	__BCinline__ static constexpr int ITERATOR() { return 0; }

	__BCinline__ operator const PARENT() const	{ return parent; }

	PARENT parent;
	array array_slice;

	__BCinline__ Array_Row(array array, PARENT parent_) : array_slice(array), parent(parent_) {}
	__BCinline__ int size() const { return parent.cols(); }
	__BCinline__ const auto inner_shape() const 			{ return l_array<1>([&](int i) { return i == 0 ? parent.cols() : 1; }); }
	__BCinline__ const auto outer_shape() const 			{ return l_array<1>([&](int i) { return i == 0 ? parent.ld1()  : 0; }); }

	__BCinline__ const auto& operator [] (int i) const { return array_slice[i * this->ld1()]; }
	__BCinline__ 	   auto& operator [] (int i)  	   { return array_slice[i * this->ld1()]; }

	void print_dimensions() 		const { parent.print_dimensions(); }
	void print_leading_dimensions()	const { parent.print_dimensions(); }

	__BCinline__ const auto slice(int i) const { return Array_Scalar<self>(&array_slice[i * this->ld1()], *this); }
	__BCinline__	   auto slice(int i) 	   { return Array_Scalar<self>(&array_slice[i * this->ld1()], *this); }

	__BCinline__ const auto& memptr() const { return *this; }
	__BCinline__	   auto& memptr()  	 	{ return *this; }

};
}
}


#endif /* TENSOR_CORE_ROWVECTOR_H_ */
