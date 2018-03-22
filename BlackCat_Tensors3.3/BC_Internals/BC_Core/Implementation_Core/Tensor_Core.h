

/*
 * Shape.h
 *
 *  Created on: Jan 18, 2018
 *      Author: joseph
 */

#ifndef SHAPE_H_
#define SHAPE_H_

#include "BC_Expressions/Expression_Base.h"
#include "Tensor_Core_Slice.h"
#include "Tensor_Core_Scalar.h"
#include "Tensor_Core_RowVector.h"
#include "../../BC_MathLibraries/Mathematics_CPU.h"
#include "Tensor_Core_Piece.h"
#include "Determiners.h"
namespace BC {

template<class T>
struct Tensor_Core : expression<_scalar<T>, Tensor_Core<T>>{

	static constexpr int RANK() { return _rankOf<T>; }
	static constexpr int LAST() { return RANK() - 1;}

	using self = Tensor_Core<T>;
	using dimlist = std::vector<int>;
	using scalar_type = _scalar<T>;
	using Mathlib = _mathlib<T>;
//	using slice_type = std::conditional_t<(RANK() == 1), Tensor_Scalar<self>, std::conditional_t<(RANK() == 0), self, Tensor_Slice<self>>>;
	using slice_type = std::conditional_t<(RANK() <= 1), Tensor_Scalar<self>, Tensor_Slice<self>>;

	scalar_type* array;
	int* is = Mathlib::unified_initialize(is, RANK());
	int* os = Mathlib::unified_initialize(os, RANK());

	operator 	   scalar_type*()       { return array; }
	operator const scalar_type*() const { return array; }

	Tensor_Core() {
		static_assert(RANK() == 0, "DEFAULT CONSTRUCTOR FOR TENSOR_CORE ONLY AVAILABLE FOR RANK == 0 (SCALAR)");
		Mathlib::initialize(array, 1);
	}

	Tensor_Core(dimlist param) {
		if (param.size() != RANK())
			throw std::invalid_argument("dimlist- rank != TENSOR_CORE::RANK");

		if (RANK() > 0) {
			Mathlib::HostToDevice(is, &param[0], RANK());

			os[0] = is[0];
			for (int i = 1; i < RANK(); ++i) {
				os[i] = os[i - 1] * is[i];
			}
		}
		Mathlib::initialize(array, size());
	}
	Tensor_Core(const int* param) {
		if (RANK() > 0) {
			Mathlib	::HostToDevice(is, &param[0], RANK());

			os[0] = is[0];
			for (int i = 1; i < RANK(); ++i) {
				os[i] = os[i - 1] * is[i];
			}
		}
		Mathlib::initialize(array, size());
	}

	__BCinline__ int rank() const { return RANK(); }
	__BCinline__ int size() const { return RANK() > 0 ? os[LAST()] : 1;    }
	__BCinline__ int rows() const { return RANK() > 0 ? is[0] : 1; }
	__BCinline__ int cols() const { return RANK() > 1 ? is[1] : 1; }
	__BCinline__ int dimension(int i) const { return RANK() > i ? is[i] : 1; }

	__BCinline__ int LD_rows() const { return RANK() > 0 ? os[0] : 1; }
	__BCinline__ int LD_cols() const { return RANK() > 1 ? os[1] : 1; }
	__BCinline__ int LDdimension(int i) const { return RANK() > i + 1 ? os[i] : 1; }
	__BCinline__	   scalar_type& operator [] (int index) 	  { return RANK() == 0 ? array[0] : array[index]; };
	__BCinline__ const scalar_type& operator [] (int index) const { return RANK() == 0 ? array[0] : array[index]; };

	__BCinline__ const auto innerShape() const { return is; }
	__BCinline__ const auto outerShape() const { return os; }

	__BCinline__ const auto slice(int i) const {  return slice_type(&array[RANK() == 1 ? i : (os[LAST() - 1] * i)], *this); }
	__BCinline__	   auto slice(int i) 	   {  return slice_type(&array[RANK() == 1 ? i : (os[LAST() - 1] * i)], *this); }

	__BCinline__ const auto scalar(int i) const { static_assert (RANK() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array[i], *this); }
	__BCinline__	   auto scalar(int i) 	    { static_assert (RANK() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array[i], *this); }
	__BCinline__ const auto row(int i) const { static_assert (RANK() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array[i], *this); }
	__BCinline__	   auto row(int i) 	     { static_assert (RANK() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array[i], *this); }
	__BCinline__ const auto col(int i) const { static_assert (RANK() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }
	__BCinline__	   auto col(int i) 	     { static_assert (RANK() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }

	const scalar_type* core() const { return array; }
		  scalar_type* core()  	    { return array; }

	void print() const { Mathlib::print(array, this->innerShape(),rank(), 4); }

	void printDimensions() const {
		for (int i = 0; i < RANK(); ++i) {
			std::cout << "[" << is[i] << "]";
		}
		std::cout << std::endl;
	}
	void printLDDimensions() const {
		for (int i = 0; i < RANK(); ++i) {
			std::cout << "[" << os[i] << "]";
		}
		std::cout << std::endl;
	}

	void resetShape(dimlist sh)  {
		os[0] = sh[0];
		is[0] = sh[0];
		for (int i = 1; i < RANK(); ++i) {
			is[i] = sh[i];
			os[i] = os[i - 1] * is[i];
		}
	}
};
}

#endif /* SHAPE_H_ */


//COMPILE TIME CHECKS MOVED TO TENSOR_BASE ->>>> ENABLES FUNCTIONAL PASSING // EASIER PROGRAMMING FOR HANDLING EXPRESSION SLICES/ROW/COLS
//__BCinline__ const auto slice(int i) const { static_assert (RANK() != 0, "SLICE OF SCALAR NOT DEFINED");
//													return slice_type(&array[RANK() == 1 ? i : (os[LAST() - 1] * i)], *this); }
//
//__BCinline__	   auto slice(int i) 	   { static_assert (RANK() != 0, "SLICE OF SCALAR NOT DEFINED");
//													return slice_type(&array[RANK() == 1 ? i : (os[LAST() - 1] * i)], *this); }
//
//__BCinline__ const auto scalar(int i) const { static_assert (RANK() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array[i], *this); }
//__BCinline__	   auto scalar(int i) 	    { static_assert (RANK() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array[i], *this); }
//__BCinline__ const auto row(int i) const { static_assert (RANK() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array[i], *this); }
//__BCinline__	   auto row(int i) 	     { static_assert (RANK() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array[i], *this); }
//__BCinline__ const auto col(int i) const { static_assert (RANK() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }
//__BCinline__	   auto col(int i) 	     { static_assert (RANK() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }


