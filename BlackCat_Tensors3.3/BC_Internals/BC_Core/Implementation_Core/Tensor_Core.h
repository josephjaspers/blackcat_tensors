

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

	__BCinline__ static constexpr int DIMS() { return _rankOf<T>; }
	__BCinline__ static constexpr int LAST() { return DIMS() - 1;}

	using self = Tensor_Core<T>;
	using dimlist = std::vector<int>;
	using scalar_type = _scalar<T>;
	using Mathlib = _mathlib<T>;
	using slice_type = std::conditional_t<(DIMS() == 0), self, Tensor_Slice<self>>;

	scalar_type* array;
	int* is = Mathlib::unified_initialize(is, DIMS());
	int* os = Mathlib::unified_initialize(os, DIMS());

	operator 	   scalar_type*()       { return array; }
	operator const scalar_type*() const { return array; }

	Tensor_Core() {
		static_assert(DIMS() == 0, "DEFAULT CONSTRUCTOR FOR TENSOR_CORE ONLY AVAILABLE FOR RANK == 0 (SCALAR)");
		Mathlib::initialize(array, 1);
	}

	Tensor_Core(dimlist param) {
		if (param.size() != DIMS())
			throw std::invalid_argument("dimlist- rank != TENSOR_CORE::RANK");

		if (DIMS() > 0) {
			Mathlib::HostToDevice(is, &param[0], DIMS());

			os[0] = is[0];
			for (int i = 1; i < DIMS(); ++i) {
				os[i] = os[i - 1] * is[i];
			}
		}
		Mathlib::initialize(array, size());
	}
	Tensor_Core(const int* param) {
		if (DIMS() > 0) {
			Mathlib	::HostToDevice(is, &param[0], DIMS());

			os[0] = is[0];
			for (int i = 1; i < DIMS(); ++i) {
				os[i] = os[i - 1] * is[i];
			}
		}
		Mathlib::initialize(array, size());
	}

	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int size() const { return DIMS() > 0 ? os[LAST()] : 1;    }
	__BCinline__ int rows() const { return DIMS() > 0 ? is[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? is[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? is[i] : 1; }

	__BCinline__ int LD_rows() const { return DIMS() > 0 ? os[0] : 1; }
	__BCinline__ int LD_cols() const { return DIMS() > 1 ? os[1] : 1; }
	__BCinline__ int LDdimension(int i) const { return DIMS() > i + 1 ? os[i] : 1; }
	__BCinline__	   scalar_type& operator [] (int index) 	  { return DIMS() == 0 ? array[0] : array[index]; };
	__BCinline__ const scalar_type& operator [] (int index) const { return DIMS() == 0 ? array[0] : array[index]; };

	__BCinline__ const auto innerShape() const { return is; }
	__BCinline__ const auto outerShape() const { return os; }

	struct slice_type_ref   {
	__BCinline__ static 		 auto foo(		 Tensor_Core& tc, int i) { return tc; };
	__BCinline__ static const auto foo(const Tensor_Core& tc, int i) { return tc; }};
	struct slice_type_slice {
	__BCinline__ static 		 auto foo(		 Tensor_Core& tc, int i) { return slice_type(&tc.array[DIMS() == 1 ? i : (tc.os[LAST() - 1] * i)], tc); };
	__BCinline__ static const auto foo(const Tensor_Core& tc, int i) { return slice_type(&tc.array[DIMS() == 1 ? i : (tc.os[LAST() - 1] * i)], tc); }};

	using slice_return = std::conditional_t<(DIMS() == 0), slice_type_ref, slice_type_slice>;

	__BCinline__ const auto slice(int i) const { return slice_return::foo(*this, i); }
	__BCinline__	   auto slice(int i) 	   { return slice_return::foo(*this, i); }

	__BCinline__ const auto scalar(int i) const { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array[i], *this); }
	__BCinline__	   auto scalar(int i) 	    { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array[i], *this); }
	__BCinline__ const auto row(int i) const { static_assert (DIMS() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array[i], *this); }
	__BCinline__	   auto row(int i) 	     { static_assert (DIMS() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array[i], *this); }
	__BCinline__ const auto col(int i) const { static_assert (DIMS() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }
	__BCinline__	   auto col(int i) 	     { static_assert (DIMS() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }

	const scalar_type* core() const { return array; }
		  scalar_type* core()  	    { return array; }

	void print() const { Mathlib::print(array, this->innerShape(),dims(), 4); }

	void printDimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << is[i] << "]";
		}
		std::cout << std::endl;
	}
	void printLDDimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << os[i] << "]";
		}
		std::cout << std::endl;
	}

	void resetShape(dimlist sh)  {
		os[0] = sh[0];
		is[0] = sh[0];
		for (int i = 1; i < DIMS(); ++i) {
			is[i] = sh[i];
			os[i] = os[i - 1] * is[i];
		}
	}
};
}

#endif /* SHAPE_H_ */


//COMPILE TIME CHECKS MOVED TO TENSOR_BASE ->>>> ENABLES FUNCTIONAL PASSING // EASIER PROGRAMMING FOR HANDLING EXPRESSION SLICES/ROW/COLS
//__BCinline__ const auto slice(int i) const { static_assert (DIMS() != 0, "SLICE OF SCALAR NOT DEFINED");
//													return slice_type(&array[DIMS() == 1 ? i : (os[LAST() - 1] * i)], *this); }
//
//__BCinline__	   auto slice(int i) 	   { static_assert (DIMS() != 0, "SLICE OF SCALAR NOT DEFINED");
//													return slice_type(&array[DIMS() == 1 ? i : (os[LAST() - 1] * i)], *this); }
//
//__BCinline__ const auto scalar(int i) const { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array[i], *this); }
//__BCinline__	   auto scalar(int i) 	    { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array[i], *this); }
//__BCinline__ const auto row(int i) const { static_assert (DIMS() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array[i], *this); }
//__BCinline__	   auto row(int i) 	     { static_assert (DIMS() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array[i], *this); }
//__BCinline__ const auto col(int i) const { static_assert (DIMS() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }
//__BCinline__	   auto col(int i) 	     { static_assert (DIMS() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }


