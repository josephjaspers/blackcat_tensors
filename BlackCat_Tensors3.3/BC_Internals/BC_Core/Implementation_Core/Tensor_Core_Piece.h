/*
 * Tensor_Slice.cu
 *
 *  Created on: Feb 25, 2018
 *      Author: joseph
 */

#ifndef TENSOR_PIECE_CU_
#define TENSOR_PIECE_CU_

#include "BC_Expressions/Expression_Base.h"
#include "Determiners.h"
#include <iostream>
#include "Tensor_Core_Scalar.h"
#include "Tensor_Core_RowVector.h"
namespace BC {


template<class PARENT, class KERNEL>
	struct Tensor_Piece {

	using scalar_type = _scalar<PARENT>;
	using self = Tensor_Piece<PARENT, KERNEL>;
	using slice_type = Tensor_Slice<self>;

	static constexpr int RANK()	 { return -1; }
	static constexpr int LAST()  { return  ((KERNEL::LAST() - 1) > 0) ? (KERNEL::LAST() - 1) : 0; }
	static_assert(KERNEL::RANK() <= PARENT::RANK(), "Kernel::RANK() must be equal or less than PARENT::RANK()");

	const PARENT parent;
	const KERNEL kernel;
	scalar_type* array_piece;

	operator 	   scalar_type*()       { return array_piece; }
	operator const scalar_type*() const { return array_piece; }

	Tensor_Piece(scalar_type* array, const PARENT& parent_) : array_piece(array), parent(parent_) {}

	__BCinline__ int dims() const { return RANK(); }
	__BCinline__ int size() const { return kernel.size();    }
	__BCinline__ int rows() const { return kernel.rows();	}
	__BCinline__ int cols() const { return kernel.cols();  }
	__BCinline__ int dimension(int i) const { return kernel.dimension(); }
	__BCinline__ int LD_rows() const { return parent.LD_rows(); }
	__BCinline__ int LD_cols() const { return parent.LD_cols(); }
	__BCinline__ int LDdimension(int i) const { return RANK() > i + 1 ? parent.outerShape()[i] : 1; }
	__BCinline__ const auto& operator [] (int i) const { return array_piece[i]; }
	__BCinline__ auto& operator [] (int i)  	       { return array_piece[i]; }

	void printDimensions() 		const { kernel.printDimensions(); }
	void printLDDimensions()	const { kernel.printDimensions(); }

	__BCinline__ const auto innerShape() const 			{ return kernel.innerShape(); }
	__BCinline__ const auto outerShape() const 			{ return parent.outerShape(); }

	const auto slice(int i) const { return Tensor_Slice<self>(&array_piece[RANK() == 1 ? i : (parent.outerShape()[LAST() - 1] * i)], *this); }
		  auto slice(int i) 	  { return Tensor_Slice<self>(&array_piece[RANK() == 1 ? i : (parent.outerShape()[LAST() - 1] * i)], *this); }

	__BCinline__ const auto scalar(int i) const { return Tensor_Scalar<self>(&array_piece[i], *this); }
	__BCinline__ auto scalar(int i) { return Tensor_Scalar<self>(&array_piece[i], *this); }

	__BCinline__ const auto row(int i) const { return Tensor_Row<self>(&array_piece[i], *this); }
	__BCinline__ auto row(int i) { return Tensor_Row<self>(&array_piece[i], *this); }
	};


}



#endif /* TENSOR_SLICE_CU_ */
