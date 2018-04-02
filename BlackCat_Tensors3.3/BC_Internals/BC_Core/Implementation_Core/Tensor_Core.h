

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
#include "Tensor_Core_Reshape.h"
#include "Tensor_Core_Interface.h"
#include "../../BC_MathLibraries/Mathematics_CPU.h"
#include "Tensor_Core_Piece.h"
#include "Determiners.h"
namespace BC {

template<class T>
struct Tensor_Core : Tensor_Core_Base<Tensor_Core<T>, _rankOf<T>>{

	__BCinline__ static constexpr int DIMS() { return _rankOf<T>; }
	__BCinline__ static constexpr int LAST() { return DIMS() - 1;}

	using self = Tensor_Core<T>;
	using scalar_type = _scalar<T>;
	using Mathlib = _mathlib<T>;
	using slice_type = Tensor_Slice<self>;

	scalar_type* array;
	int* is = Mathlib::unified_initialize(is, DIMS());
	int* os = Mathlib::unified_initialize(os, DIMS());

	operator 	   scalar_type*()       { return array; }
	operator const scalar_type*() const { return array; }

	Tensor_Core() = default;
	Tensor_Core(const Tensor_Core&) = default;
	Tensor_Core(Tensor_Core&&) = default;

	Tensor_Core(std::vector<int> param) {

		if (DIMS() > 0) {
			is[0] = param[0];
			os[0] = is[0];
			for (int i = 1; i < DIMS(); ++i) {
				is[i] = param[i];
				os[i] = os[i - 1] * is[i];
			}
		}
		Mathlib::initialize(array, this->size());
	}
	Tensor_Core(const int* param) {
		if (DIMS() > 0) {
			Mathlib::HostToDevice(is, &param[0], DIMS());

			os[0] = is[0];
			for (int i = 1; i < DIMS(); ++i) {
				os[i] = os[i - 1] * is[i];
			}
		}
		Mathlib::initialize(array, this->size());
	}
	__BCinline__	   scalar_type& operator [] (int index) 	  { return DIMS() == 0 ? array[0] : array[index]; };
	__BCinline__ const scalar_type& operator [] (int index) const { return DIMS() == 0 ? array[0] : array[index]; };

	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int size() const { return DIMS() > 0 ? os[LAST()] : 1;    }
	__BCinline__ int rows() const { return DIMS() > 0 ? is[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? is[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? is[i] : 1; }

	__BCinline__ int LD_rows() const { return DIMS() > 0 ? os[0] : 1; }
	__BCinline__ int LD_cols() const { return DIMS() > 1 ? os[1] : 1; }
	__BCinline__ int LD_dimension(int i) const { return DIMS() > i + 1 ? os[i] : 1; }

	__BCinline__ const auto innerShape() const { return is; }
	__BCinline__ const auto outerShape() const { return os; }

	__BCinline__
	__BCinline__ const auto slice(int i) const { return slice_type(&array[slice_index(i)],*this); }
	__BCinline__	   auto slice(int i) 	   { return slice_type(&array[slice_index(i)],*this); }

	__BCinline__ const auto scalar(int i) const { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array[i], *this); }
	__BCinline__	   auto scalar(int i) 	    { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return Tensor_Scalar<self>(&array[i], *this); }
	__BCinline__ const auto row(int i) const { static_assert (DIMS() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array[i], *this); }
	__BCinline__	   auto row(int i) 	     { static_assert (DIMS() != 2, "ROW OF NON-MATRIX NOT DEFINED"); return Tensor_Row<self>(&array[i], *this); }
	__BCinline__ const auto col(int i) const { static_assert (DIMS() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }
	__BCinline__	   auto col(int i) 	     { static_assert (DIMS() != 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }

	template<class... integers>  const auto reshape(integers... ints) const {
		return Tensor_Reshape<const self, sizeof...(integers)>(*this, ints...); }
	template<class... integers>   	   auto reshape(integers... ints)  		{
		return Tensor_Reshape<self, sizeof...(integers)>(*this, ints...); }

	const scalar_type* getIterator() const { return array; }
		  scalar_type* getIterator()  	    { return array; }


	template<class... integers, int dim = 0>
	void resetShape(integers... ints)  {
		this->init<0>(ints...);
		Mathlib::destroy(array);
		Mathlib::initialize(array, this->size());
	}


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


	int slice_index(int i) const {
		if (DIMS() == 0)
			return 0;
		else if (DIMS() == 1)
			return i;
		else
			return outerShape()[LAST() - 1] * i;
	}


	template<int d>
	void init() {/*I AM FRIEND I HELP COMPILE DONT DELETE ME */}

	template<int dim, class... integers> __BCinline__
	void init(int front, integers... ints) {
		is[dim] = front;
		if (dim > 0)
			os[dim] = front * os[dim - 1];
		else
			os[0] = front;

		if (dim != DIMS() - 1) {
			init<(dim + 1 < DIMS() ? dim + 1 : DIMS())>(ints...);
		}
	}



};
}

#endif /* SHAPE_H_ */
