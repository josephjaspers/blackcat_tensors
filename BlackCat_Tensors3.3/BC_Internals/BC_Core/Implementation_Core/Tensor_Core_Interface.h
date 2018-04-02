/*
 * Tensor_Core_Interface.h
 *
 *  Created on: Apr 1, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CORE_INTERFACE_H_
#define TENSOR_CORE_INTERFACE_H_

#include "BlackCat_Internal_Definitions.h"
#include "Determiners.h"

namespace BC {

template<class derived, int DIMENSION>
struct Tensor_Core_Base {

	static constexpr int DIMS() { return DIMENSION; }
	static constexpr int LAST() { return DIMENSION - 1; }

	using self = derived;
	using slice_type = std::conditional_t<DIMS() == 0, self, Tensor_Slice<self>>;

private:
	const derived& base() const { return static_cast<const derived&>(*this); }
		  derived& base() 		{ return static_cast<	   derived&>(*this); }

	__BCinline__ const auto array() const { return base().getIterator(); };
	__BCinline__	   auto array() 	  { return base().getIterator(); };

public:

	__BCinline__ 	   auto& operator [] (int index) 	   { return DIMS() == 0 ? array()[0] : array()[index]; };
	__BCinline__ const auto& operator [] (int index) const { return DIMS() == 0 ? array()[0] : array()[index]; };

	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int size() const { return DIMS() > 0 ? outerShape()[LAST()] : 1;    }
	__BCinline__ int rows() const { return DIMS() > 0 ? innerShape()[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? innerShape()[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? base().innerShape()[i] : 1; }

	__BCinline__ int LD_rows() const { return DIMS() > 0 ? outerShape()[0] : 1; }
	__BCinline__ int LD_cols() const { return DIMS() > 1 ? outerShape()[1] : 1; }
	__BCinline__ int LD_dimension(int i) const { return DIMS() > i + 1 ? outerShape()[i] : 1; }

	__BCinline__ const auto innerShape() const { return base().innerShape(); }
	__BCinline__ const auto outerShape() const { return base().outerShape(); }


//	__BCinline__ const auto slice(int i) const { return slice_type(&array()[slice_index(i)],*this); }
//	__BCinline__	   auto slice(int i) 	   { return slice_type(&array()[slice_index(i)],*this); }
//
//	__BCinline__ const auto scalar(int i) const { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array()[i], *this); }
//	__BCinline__	   auto scalar(int i) 	    { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array()[i], *this); }
//	__BCinline__ const auto row(int i) const { static_assert (DIMS() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array()[i], *this); }
//	__BCinline__	   auto row(int i) 	     { static_assert (DIMS() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array()[i], *this); }
//	__BCinline__ const auto col(int i) const { static_assert (DIMS() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }
//	__BCinline__	   auto col(int i) 	     { static_assert (DIMS() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }


//	int slice_index(int i) const {
//		if (DIMS() == 0)
//			return 0;
//		else if (DIMS() == 1)
//			return i;
//		else
//			return base().outerShape()[LAST() - 1] * i;
//	}
};

}


#endif /* TENSOR_CORE_INTERFACE_H_ */
